// High-performance C++17 / OpenMP native kernels for PINPOINT pixel clustering,
// multi-radius neighborhood search, 32x32 bilinear micro-patch rasterization,
// and explicit micron-scale 3D multi-tracklet / Adaptive Primary-Secondary Vertex / Kink reconstruction.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

inline uint64_t pack_grid_cell(int64_t gx, int64_t gy) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(gx)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(gy));
}

struct MinRootDSU {
    std::vector<int32_t> parent;
    explicit MinRootDSU(int32_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int32_t find(int32_t x) {
        int32_t r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) {
            int32_t nxt = parent[x];
            parent[x] = r;
            x = nxt;
        }
        return r;
    }
    void unite(int32_t a, int32_t b) {
        int32_t ra = find(a);
        int32_t rb = find(b);
        if (ra != rb) {
            if (ra < rb) parent[rb] = ra;
            else parent[ra] = rb;
        }
    }
};

struct Tracklet3D {
    double x0;     // transverse x at z0
    double y0;     // transverse y at z0
    double z0;     // upstream plane z
    double tx;     // dx/dz slope
    double ty;     // dy/dz slope
    double sigma2; // effective transverse position variance at vertex z (mm^2)
    double chi2;
    int n_hits;
    int first_plane;
    int first_cl;
    int last_plane;
    int last_cl;
};

// Solve symmetric 3x3 positive-definite linear system A * v = b via Cholesky / Cramer with safe fallback
bool solve_sym_3x3(const double A[3][3], const double b[3], double out[3]) {
    const double det =
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (!std::isfinite(det) || std::abs(det) < 1e-18) {
        return false;
    }
    const double inv_det = 1.0 / det;
    double invA[3][3];
    invA[0][0] =  (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    invA[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * inv_det;
    invA[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
    invA[1][0] = invA[0][1];
    invA[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    invA[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) * inv_det;
    invA[2][0] = invA[0][2];
    invA[2][1] = invA[1][2];
    invA[2][2] =  (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

    for (int i = 0; i < 3; ++i) {
        out[i] = invA[i][0] * b[0] + invA[i][1] * b[1] + invA[i][2] * b[2];
        if (!std::isfinite(out[i])) return false;
    }
    return true;
}

bool fit_weighted_lines_vertex(
    const std::vector<Tracklet3D>& trks,
    const std::vector<double>& weights,
    double prior_x,
    double prior_y,
    double prior_z,
    double prior_xy_prec,
    double prior_z_prec,
    double v_out[3]
) {
    // Minimize sum_i w_i * [(x0_i + tx_i*(zv - z0_i) - xv)^2 + (y0_i + ty_i*(zv - z0_i) - yv)^2]
    //        + prior_xy_prec * [(xv - prior_x)^2 + (yv - prior_y)^2] + prior_z_prec * (zv - prior_z)^2
    double A[3][3] = {{prior_xy_prec, 0.0, 0.0}, {0.0, prior_xy_prec, 0.0}, {0.0, 0.0, prior_z_prec}};
    double rhs[3] = {prior_xy_prec * prior_x, prior_xy_prec * prior_y, prior_z_prec * prior_z};

    for (std::size_t i = 0; i < trks.size(); ++i) {
        const double w = weights[i];
        if (w <= 1e-12) continue;
        const auto& t = trks[i];
        const double ax = t.x0 - t.tx * t.z0;
        const double ay = t.y0 - t.ty * t.z0;

        A[0][0] += w;
        A[0][2] -= w * t.tx;
        A[2][0] -= w * t.tx;

        A[1][1] += w;
        A[1][2] -= w * t.ty;
        A[2][1] -= w * t.ty;

        A[2][2] += w * (t.tx * t.tx + t.ty * t.ty);

        rhs[0] += w * ax;
        rhs[1] += w * ay;
        rhs[2] -= w * (t.tx * ax + t.ty * ay);
    }
    return solve_sym_3x3(A, rhs, v_out);
}

}  // namespace

extern "C" {

int32_t fast_cluster_pixel_hits(
    const uint64_t* cid,
    const double* xyz,
    int64_t n_hits,
    int64_t nx,
    int64_t ny,
    double radius_mm,
    int32_t* out_hit_order,
    int32_t* out_cluster_offsets,
    int16_t* out_cluster_plane,
    uint64_t* out_cluster_first_cid,
    int32_t* out_hit_to_cluster
) {
    if (n_hits <= 0) {
        out_cluster_offsets[0] = 0;
        return 0;
    }

    std::vector<int32_t> canon(n_hits);
    std::iota(canon.begin(), canon.end(), 0);
    std::stable_sort(canon.begin(), canon.end(), [&](int32_t a, int32_t b) {
        return cid[a] < cid[b];
    });

    const uint64_t stride = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
    const double cell_size = std::max(radius_mm, 1e-6);
    const double inv_cell = 1.0 / cell_size;

    std::vector<int32_t> hit_to_cl_sorted(n_hits);
    std::vector<int32_t> cl_counts;
    cl_counts.reserve(std::min<int64_t>(n_hits, 4096));

    int32_t cluster_offset_id = 0;
    int64_t start = 0;

    std::unordered_map<uint64_t, int32_t> cell_head;
    cell_head.reserve(2048);

    while (start < n_hits) {
        const uint64_t plane_u = cid[canon[start]] / stride;
        int64_t stop = start + 1;
        while (stop < n_hits && (cid[canon[stop]] / stride) == plane_u) {
            ++stop;
        }
        const int32_t m = static_cast<int32_t>(stop - start);
        const int16_t plane_val = static_cast<int16_t>(plane_u);

        if (m == 1) {
            hit_to_cl_sorted[start] = cluster_offset_id;
            out_cluster_plane[cluster_offset_id] = plane_val;
            out_cluster_first_cid[cluster_offset_id] = cid[canon[start]];
            cl_counts.push_back(1);
            ++cluster_offset_id;
            start = stop;
            continue;
        }

        MinRootDSU dsu(m);
        cell_head.clear();
        std::vector<int32_t> next_in_cell(m, -1);

        for (int32_t i = 0; i < m; ++i) {
            const int32_t orig_i = canon[start + i];
            const double xi = xyz[3 * orig_i + 0];
            const double yi = xyz[3 * orig_i + 1];
            const int64_t gx = static_cast<int64_t>(std::floor(xi * inv_cell));
            const int64_t gy = static_cast<int64_t>(std::floor(yi * inv_cell));

            for (int64_t dx_c = -1; dx_c <= 1; ++dx_c) {
                for (int64_t dy_c = -1; dy_c <= 1; ++dy_c) {
                    const uint64_t key = pack_grid_cell(gx + dx_c, gy + dy_c);
                    auto it = cell_head.find(key);
                    if (it == cell_head.end()) continue;
                    for (int32_t j = it->second; j >= 0; j = next_in_cell[j]) {
                        const int32_t orig_j = canon[start + j];
                        const double dx = xi - xyz[3 * orig_j + 0];
                        const double dy = yi - xyz[3 * orig_j + 1];
                        // Match scipy.spatial.cKDTree.query_pairs Euclidean predicate exactly
                        if (std::sqrt(dx * dx + dy * dy) <= radius_mm) {
                            dsu.unite(i, j);
                        }
                    }
                }
            }
            const uint64_t self_key = pack_grid_cell(gx, gy);
            auto [it_ins, inserted] = cell_head.emplace(self_key, i);
            if (!inserted) {
                next_in_cell[i] = it_ins->second;
                it_ins->second = i;
            }
        }

        std::vector<int32_t> root_to_comp(m, -1);
        int32_t n_comp = 0;
        for (int32_t i = 0; i < m; ++i) {
            const int32_t r = dsu.find(i);
            int32_t comp = root_to_comp[r];
            if (comp < 0) {
                comp = n_comp++;
                root_to_comp[r] = comp;
                out_cluster_plane[cluster_offset_id + comp] = plane_val;
                out_cluster_first_cid[cluster_offset_id + comp] = cid[canon[start + r]];
                cl_counts.push_back(0);
            }
            hit_to_cl_sorted[start + i] = cluster_offset_id + comp;
            cl_counts[cluster_offset_id + comp] += 1;
        }
        cluster_offset_id += n_comp;
        start = stop;
    }

    const int32_t total_clusters = cluster_offset_id;
    out_cluster_offsets[0] = 0;
    for (int32_t c = 0; c < total_clusters; ++c) {
        out_cluster_offsets[c + 1] = out_cluster_offsets[c] + cl_counts[c];
    }

    // Counting-sort canon[0..n_hits-1] into out_hit_order preserving ascending cid within each cluster
    std::vector<int32_t> write_cursor(out_cluster_offsets, out_cluster_offsets + total_clusters);
    for (int64_t k = 0; k < n_hits; ++k) {
        const int32_t cl_id = hit_to_cl_sorted[k];
        const int32_t orig_idx = canon[k];
        out_hit_order[write_cursor[cl_id]++] = orig_idx;
        out_hit_to_cluster[orig_idx] = cl_id;
    }
    return total_clusters;
}

void fast_plane_neighborhoods(
    const int64_t* cl_plane,
    const double* x_mean,
    const double* y_mean,
    const double* n_hits,
    int64_t n_cl,
    double* cnt_r015,
    double* hits_r015,
    double* cnt_r040,
    double* hits_r040,
    double* cnt_r1,
    double* cnt_r5,
    double* hits_r5
) {
    std::memset(cnt_r015, 0, sizeof(double) * n_cl);
    std::memset(hits_r015, 0, sizeof(double) * n_cl);
    std::memset(cnt_r040, 0, sizeof(double) * n_cl);
    std::memset(hits_r040, 0, sizeof(double) * n_cl);
    std::memset(cnt_r1, 0, sizeof(double) * n_cl);
    std::memset(cnt_r5, 0, sizeof(double) * n_cl);
    std::memset(hits_r5, 0, sizeof(double) * n_cl);
    if (n_cl <= 1) return;

    // Group cluster indices by plane
    int64_t start = 0;
    std::unordered_map<uint64_t, int32_t> cell_head;
    while (start < n_cl) {
        const int64_t p = cl_plane[start];
        int64_t stop = start + 1;
        while (stop < n_cl && cl_plane[stop] == p) ++stop;
        const int32_t m = static_cast<int32_t>(stop - start);
        if (m <= 1) {
            start = stop;
            continue;
        }

        if (m <= 96) {
            for (int32_t i = 0; i < m; ++i) {
                const int64_t gi = start + i;
                const double xi = x_mean[gi];
                const double yi = y_mean[gi];
                const double wi = n_hits[gi];
                for (int32_t j = i + 1; j < m; ++j) {
                    const int64_t gj = start + j;
                    const double dx = xi - x_mean[gj];
                    if (std::abs(dx) > 5.0) continue;
                    const double dy = yi - y_mean[gj];
                    if (std::abs(dy) > 5.0) continue;
                    const double d = std::sqrt(dx * dx + dy * dy);
                    if (d <= 5.0) {
                        const double wj = n_hits[gj];
                        cnt_r5[gi] += 1.0;
                        cnt_r5[gj] += 1.0;
                        hits_r5[gi] += wj;
                        hits_r5[gj] += wi;
                        if (d <= 1.0) {
                            cnt_r1[gi] += 1.0;
                            cnt_r1[gj] += 1.0;
                            if (d <= 0.40) {
                                cnt_r040[gi] += 1.0;
                                cnt_r040[gj] += 1.0;
                                hits_r040[gi] += wj;
                                hits_r040[gj] += wi;
                                if (d <= 0.15) {
                                    cnt_r015[gi] += 1.0;
                                    cnt_r015[gj] += 1.0;
                                    hits_r015[gi] += wj;
                                    hits_r015[gj] += wi;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            constexpr double inv_cell = 1.0 / 5.0;
            cell_head.clear();
            cell_head.reserve(m * 2);
            std::vector<int32_t> next_in_cell(m, -1);
            for (int32_t i = 0; i < m; ++i) {
                const int64_t gi = start + i;
                const double xi = x_mean[gi];
                const double yi = y_mean[gi];
                const double wi = n_hits[gi];
                const int64_t gx = static_cast<int64_t>(std::floor(xi * inv_cell));
                const int64_t gy = static_cast<int64_t>(std::floor(yi * inv_cell));

                for (int64_t dxc = -1; dxc <= 1; ++dxc) {
                    for (int64_t dyc = -1; dyc <= 1; ++dyc) {
                        auto it = cell_head.find(pack_grid_cell(gx + dxc, gy + dyc));
                        if (it == cell_head.end()) continue;
                        for (int32_t j = it->second; j >= 0; j = next_in_cell[j]) {
                            const int64_t gj = start + j;
                            const double dx = xi - x_mean[gj];
                            const double dy = yi - y_mean[gj];
                            const double d = std::sqrt(dx * dx + dy * dy);
                            if (d <= 5.0) {
                                const double wj = n_hits[gj];
                                cnt_r5[gi] += 1.0;
                                cnt_r5[gj] += 1.0;
                                hits_r5[gi] += wj;
                                hits_r5[gj] += wi;
                                if (d <= 1.0) {
                                    cnt_r1[gi] += 1.0;
                                    cnt_r1[gj] += 1.0;
                                    if (d <= 0.40) {
                                        cnt_r040[gi] += 1.0;
                                        cnt_r040[gj] += 1.0;
                                        hits_r040[gi] += wj;
                                        hits_r040[gj] += wi;
                                        if (d <= 0.15) {
                                            cnt_r015[gi] += 1.0;
                                            cnt_r015[gj] += 1.0;
                                            hits_r015[gi] += wj;
                                            hits_r015[gj] += wi;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                const uint64_t self_key = pack_grid_cell(gx, gy);
                auto [it_ins, inserted] = cell_head.emplace(self_key, i);
                if (!inserted) {
                    next_in_cell[i] = it_ins->second;
                    it_ins->second = i;
                }
            }
        }
        start = stop;
    }
}

void fast_rasterize_patches(
    const int32_t* hit_order,
    const int32_t* offsets,
    const double* pxyz,
    const double* x_mean,
    const double* y_mean,
    const int64_t* sel_local,
    const int64_t* keep_idx,
    int64_t n_patches,
    int32_t patch_size,
    double pitch_x,
    double pitch_y,
    float* out_patches
) {
    if (n_patches <= 0) return;
    const int64_t plane_elems = static_cast<int64_t>(patch_size) * static_cast<int64_t>(patch_size);
    std::memset(out_patches, 0, sizeof(float) * n_patches * 2 * plane_elems);

    const double half_center = 0.5 * (patch_size - 1);
    const double inv_px = 1.0 / pitch_x;
    const double inv_py = 1.0 / pitch_y;
    const double max_clip = static_cast<double>(patch_size) - 1.0 - 1e-5;

    for (int64_t p_i = 0; p_i < n_patches; ++p_i) {
        const int64_t cl_id = keep_idx[sel_local[p_i]];
        const int32_t s = offsets[cl_id];
        const int32_t e = offsets[cl_id + 1];
        const double xm = x_mean[cl_id];
        const double ym = y_mean[cl_id];

        float* ch0 = out_patches + (p_i * 2 + 0) * plane_elems;
        float* ch1 = out_patches + (p_i * 2 + 1) * plane_elems;
        int32_t n_out = 0;

        for (int32_t h = s; h < e; ++h) {
            const int32_t idx = hit_order[h];
            const double u = half_center + (pxyz[3 * idx + 0] - xm) * inv_px;
            const double v = half_center + (pxyz[3 * idx + 1] - ym) * inv_py;
            if (u >= -0.5 && u <= (patch_size - 0.5) && v >= -0.5 && v <= (patch_size - 0.5)) {
                const double ui = std::clamp(u, 0.0, max_clip);
                const double vi = std::clamp(v, 0.0, max_clip);
                const int32_t u0 = static_cast<int32_t>(std::floor(ui));
                const int32_t v0 = static_cast<int32_t>(std::floor(vi));
                const float du = static_cast<float>(ui - u0);
                const float dv = static_cast<float>(vi - v0);
                const int32_t u1 = std::min(u0 + 1, patch_size - 1);
                const int32_t v1 = std::min(v0 + 1, patch_size - 1);
                ch0[v0 * patch_size + u0] += (1.0f - dv) * (1.0f - du);
                ch0[v0 * patch_size + u1] += (1.0f - dv) * du;
                ch0[v1 * patch_size + u0] += dv * (1.0f - du);
                ch0[v1 * patch_size + u1] += dv * du;
            } else {
                ++n_out;
            }
        }

        if (n_out > 0) {
            const float edge_val = static_cast<float>(std::log1p(static_cast<double>(n_out)) / 5.0);
            for (int32_t r = 0; r < 2; ++r) {
                for (int32_t c = 0; c < patch_size; ++c) {
                    ch1[r * patch_size + c] = edge_val;
                    ch1[(patch_size - 1 - r) * patch_size + c] = edge_val;
                }
            }
            for (int32_t r = 0; r < patch_size; ++r) {
                ch1[r * patch_size + 0] = edge_val;
                ch1[r * patch_size + 1] = edge_val;
                ch1[r * patch_size + (patch_size - 2)] = edge_val;
                ch1[r * patch_size + (patch_size - 1)] = edge_val;
            }
        }
    }
}

// Explicit micron-scale 3D multi-tracklet, Adaptive Primary/Secondary Vertex (AVR),
// Impact Parameter Significance (d0 / sigma_d0), Two-Vertex Delta-Chi2, and 1-Prong Kink Reconstructor.
void fast_reconstruct_micron_vertex(
    const int64_t* cl_plane,
    const double* x_mean,
    const double* y_mean,
    const double* z_cl,
    const double* n_hits,
    const double* lam2,
    int64_t n_cl,
    int32_t n_pix_planes,
    double pitch_x,
    double pitch_y,
    const double* cal_start_4,    // [z, x, y, valid]
    const double* default_center, // [x_c, y_c, z_mid]
    double* out_pv_xyz,           // [3]
    int32_t* out_pv_valid,
    float* out_feat               // [16] reflection & permutation invariant features
) {
    std::memset(out_feat, 0, sizeof(float) * 16);
    *out_pv_valid = 0;

    if (cal_start_4[3] > 0.5) {
        out_pv_xyz[0] = cal_start_4[1];
        out_pv_xyz[1] = cal_start_4[2];
        out_pv_xyz[2] = cal_start_4[0];
    } else {
        out_pv_xyz[0] = default_center[0];
        out_pv_xyz[1] = default_center[1];
        out_pv_xyz[2] = default_center[2];
    }

    if (n_cl < 2 || n_pix_planes <= 0) {
        return;
    }

    // 1. Partition clusters by plane and find upstream active interaction plane p0
    // anchored near cal_start_4[0] when available (prevents locking onto downstream IPT planes on Fortune events
    // or isolated upstream albedo back-scatter).
    std::vector<std::vector<int32_t>> plane_cls(n_pix_planes);
    std::vector<double> plane_z(n_pix_planes, 0.0);
    for (int64_t i = 0; i < n_cl; ++i) {
        const int32_t p = static_cast<int32_t>(cl_plane[i]);
        if (p >= 0 && p < n_pix_planes) {
            if (plane_cls[p].empty()) plane_z[p] = z_cl[i];
            plane_cls[p].push_back(static_cast<int32_t>(i));
        }
    }

    const bool has_cal = (cal_start_4[3] > 0.5);
    const double z_cal = cal_start_4[0];
    int32_t p0 = -1;
    // Target silicon planes are 0..min(11, n_pix_planes-1); require consistency with cal_start within [-45, +55] mm
    const int32_t max_target_p = std::min(n_pix_planes, 12);
    for (int32_t p = 0; p < max_target_p; ++p) {
        if (plane_cls[p].size() >= 3) {
            if (!has_cal || (plane_z[p] >= z_cal - 45.0 && plane_z[p] <= z_cal + 165.0)) {
                p0 = p;
                break;
            }
        }
    }
    if (p0 < 0) {
        for (int32_t p = 0; p < max_target_p; ++p) {
            if (plane_cls[p].size() >= 2) {
                if (!has_cal || (plane_z[p] >= z_cal - 45.0 && plane_z[p] <= z_cal + 165.0)) {
                    p0 = p;
                    break;
                }
            }
        }
    }
    if (p0 < 0 && !has_cal) {
        for (int32_t p = 0; p < n_pix_planes; ++p) {
            if (!plane_cls[p].empty()) {
                p0 = p;
                break;
            }
        }
    }
    if (p0 < 0) return;

    // Preliminary transverse core on p0 to test whether p0-1 has a collinear parent tau stub (within 2.0 mm!)
    double x0_med = 0.0, y0_med = 0.0, w0_s = 0.0;
    for (int32_t idx : plane_cls[p0]) {
        x0_med += n_hits[idx] * x_mean[idx];
        y0_med += n_hits[idx] * y_mean[idx];
        w0_s += n_hits[idx];
    }
    if (w0_s > 0.0) { x0_med /= w0_s; y0_med /= w0_s; }
    if (p0 > 0 && !plane_cls[p0 - 1].empty() && plane_cls[p0 - 1].size() <= 3 &&
        (plane_z[p0] - plane_z[p0 - 1]) <= 36.0) {
        bool has_collinear_stub = false;
        for (int32_t idx : plane_cls[p0 - 1]) {
            if (std::hypot(x_mean[idx] - x0_med, y_mean[idx] - y0_med) <= 2.0) {
                has_collinear_stub = true;
                break;
            }
        }
        if (has_collinear_stub && (!has_cal || plane_z[p0 - 1] >= z_cal - 45.0)) {
            p0 = p0 - 1;
        }
    }

    // Compute robust core center across adjacent 30mm-spaced planes starting at p0
    int32_t p_end = p0;
    while (p_end + 1 < n_pix_planes && p_end < p0 + 4) {
        // Do not bridge across 150 mm (20 X0) FortuneBlock tungsten stacks for micron vertexing
        if (!plane_cls[p_end + 1].empty() && !plane_cls[p_end].empty() &&
            (plane_z[p_end + 1] - plane_z[p0]) > 105.0) {
            break;
        }
        ++p_end;
    }

    double w_sum = 0.0, xc_win = 0.0, yc_win = 0.0;
    for (int32_t p = p0; p <= p_end; ++p) {
        for (int32_t idx : plane_cls[p]) {
            if (has_cal && std::hypot(x_mean[idx] - cal_start_4[1], y_mean[idx] - cal_start_4[2]) > 45.0) {
                continue;
            }
            const double w = n_hits[idx];
            xc_win += w * x_mean[idx];
            yc_win += w * y_mean[idx];
            w_sum += w;
        }
    }
    if (w_sum <= 0.0) {
        for (int32_t p = p0; p <= p_end; ++p) {
            for (int32_t idx : plane_cls[p]) {
                const double w = n_hits[idx];
                xc_win += w * x_mean[idx];
                yc_win += w * y_mean[idx];
                w_sum += w;
            }
        }
    }
    if (w_sum <= 0.0) return;
    xc_win /= w_sum;
    yc_win /= w_sum;

    // First-plane MIP prong purity (distinguishes 1p/3p hadronic tau from immediate 1.4 X0 nu_e EM cascades)
    int32_t p0_mip_cnt = 0, p0_core_cnt = 0;
    for (int32_t idx : plane_cls[p0]) {
        if (std::hypot(x_mean[idx] - xc_win, y_mean[idx] - yc_win) <= 5.0) {
            ++p0_core_cnt;
            if (n_hits[idx] <= 4.0) ++p0_mip_cnt;
        }
    }
    const float p0_mip_frac = (p0_core_cnt > 0)
        ? static_cast<float>(p0_mip_cnt) / static_cast<float>(p0_core_cnt)
        : 0.0f;

    constexpr std::size_t MAX_CL_PER_PLANE = 40;
    for (int32_t p = p0; p <= p_end; ++p) {
        auto& vec = plane_cls[p];
        // Filter to 12 mm core radius around vertex axis and MIP-like clusters (n_hits <= 16)
        std::vector<int32_t> core_vec;
        core_vec.reserve(vec.size());
        for (int32_t idx : vec) {
            if (std::hypot(x_mean[idx] - xc_win, y_mean[idx] - yc_win) <= 12.0 && n_hits[idx] <= 16.0) {
                core_vec.push_back(idx);
            }
        }
        if (core_vec.size() >= 2) {
            vec.swap(core_vec);
        }
        if (vec.size() > MAX_CL_PER_PLANE) {
            std::stable_sort(vec.begin(), vec.end(), [&](int32_t a, int32_t b) {
                const double da2 = (x_mean[a] - xc_win) * (x_mean[a] - xc_win) +
                                   (y_mean[a] - yc_win) * (y_mean[a] - yc_win);
                const double db2 = (x_mean[b] - xc_win) * (x_mean[b] - xc_win) +
                                   (y_mean[b] - yc_win) * (y_mean[b] - yc_win);
                return da2 < db2;
            });
            vec.resize(MAX_CL_PER_PLANE);
        }
    }

    // 2. Seeded 3-plane triplets and 2-plane doublets across adjacent 30mm Pinpoint planes
    const double sig_hit_base2 = 0.5 * ((pitch_x * pitch_x + pitch_y * pitch_y) / 12.0) + 1e-5; // ~3.8e-5 mm^2 (~6.2 um)^2
    constexpr double kMaxSlope = 0.22;         // 220 mrad forward cone
    constexpr double kThetaMS = 1.4e-3;        // ~1.4 mrad typical stiff prong MS per block

    struct CandTrk {
        Tracklet3D trk;
        int c0, c1, c2;
    };
    std::vector<CandTrk> cands;
    cands.reserve(256);

    for (int32_t pa = p0; pa <= p_end - 1; ++pa) {
        if (plane_cls[pa].empty()) continue;
        for (int32_t pb = pa + 1; pb <= std::min(p_end, pa + 2); ++pb) {
            if (plane_cls[pb].empty()) continue;
            for (int32_t ia : plane_cls[pa]) {
                const double xa = x_mean[ia], ya = y_mean[ia], za = z_cl[ia];
                const double sa2 = sig_hit_base2 + 0.15 * lam2[ia];
                for (int32_t ib : plane_cls[pb]) {
                    const double dz_ab = z_cl[ib] - za;
                    if (dz_ab < 5.0 || dz_ab > 65.0) continue;
                    const double tx_ab = (x_mean[ib] - xa) / dz_ab;
                    const double ty_ab = (y_mean[ib] - ya) / dz_ab;
                    if (std::hypot(tx_ab, ty_ab) > kMaxSlope) continue;

                    bool found_triplet = false;
                    for (int32_t pc = pb + 1; pc <= std::min(p_end, pb + 2); ++pc) {
                        if (plane_cls[pc].empty()) continue;
                        for (int32_t ic : plane_cls[pc]) {
                            const double dz_ac = z_cl[ic] - za;
                            if (dz_ac < 15.0 || dz_ac > 95.0) continue;
                            const double tx = (x_mean[ic] - xa) / dz_ac;
                            const double ty = (y_mean[ic] - ya) / dz_ac;
                            if (std::hypot(tx, ty) > kMaxSlope) continue;

                            const double xb_pred = xa + tx * dz_ab;
                            const double yb_pred = ya + ty * dz_ab;
                            const double rx = x_mean[ib] - xb_pred;
                            const double ry = y_mean[ib] - yb_pred;
                            const double sig_mid2 = 1.5 * sig_hit_base2 + 0.15 * lam2[ib] +
                                                    (kThetaMS * dz_ab) * (kThetaMS * dz_ab);
                            const double chi2 = (rx * rx + ry * ry) / sig_mid2;
                            if (chi2 <= 16.0) {
                                const double zm = (za + z_cl[ib] + z_cl[ic]) / 3.0;
                                const double xm = (xa + x_mean[ib] + x_mean[ic]) / 3.0;
                                const double ym = (ya + y_mean[ib] + y_mean[ic]) / 3.0;
                                const double szz = (za - zm) * (za - zm) +
                                                   (z_cl[ib] - zm) * (z_cl[ib] - zm) +
                                                   (z_cl[ic] - zm) * (z_cl[ic] - zm);
                                const double tx_fit = ((za - zm) * (xa - xm) +
                                                       (z_cl[ib] - zm) * (x_mean[ib] - xm) +
                                                       (z_cl[ic] - zm) * (x_mean[ic] - xm)) / szz;
                                const double ty_fit = ((za - zm) * (ya - ym) +
                                                       (z_cl[ib] - zm) * (y_mean[ib] - ym) +
                                                       (z_cl[ic] - zm) * (y_mean[ic] - ym)) / szz;
                                const double x0_fit = xm + tx_fit * (za - zm);
                                const double y0_fit = ym + ty_fit * (za - zm);
                                const double eff_sig2 = sa2 + (0.8 * kThetaMS * 15.0) * (0.8 * kThetaMS * 15.0);
                                cands.push_back({
                                    {x0_fit, y0_fit, za, tx_fit, ty_fit, eff_sig2, chi2, 3, pa, ia, pc, ic},
                                    ia, ib, ic
                                });
                                found_triplet = true;
                            }
                        }
                    }

                    if (!found_triplet && dz_ab <= 36.0 && std::hypot(tx_ab, ty_ab) <= 0.15) {
                        const double eff_sig2 = 2.0 * sa2 + (1.6 * kThetaMS * 18.0) * (1.6 * kThetaMS * 18.0);
                        const double pseudo_chi2 = 4.0 + 200.0 * (tx_ab * tx_ab + ty_ab * ty_ab);
                        cands.push_back({
                            {xa, ya, za, tx_ab, ty_ab, eff_sig2, pseudo_chi2, 2, pa, ia, pb, ib},
                            ia, ib, -1
                        });
                    }
                }
            }
        }
    }

    // Physical Pinpoint_G4 DetectorConstruction.cc slab placement relative to PixelSensor[p0]:
    // - Inside PinpointBlock (30.01 mm): PixelSensor at -7.000, W_back at +0.505 (+7.505 from Si),
    //   W_front at +10.505 (+17.505 from Si).
    // - Therefore for p0 in {1..5}, the upstream PinpointBlock (p0-1) has W plates at
    //   z_pix[p0] - 22.505 mm and z_pix[p0] - 12.505 mm (midpoint: z_pix[p0] - 17.505 mm).
    // - For p0 == 0, PixelSensor[0] (-2991.995 mm) is UPSTREAM of W0 (-2984.49 mm = z_pix[0] + 7.505 mm),
    //   so any interaction starting at W0 has prior_z = z_pix[0] + 7.505 mm (never upstream air!).
    // - For p0 >= 6 (intermediate Pinpoint planes immediately after a 120 mm FortuneBlock),
    //   upstream FortuneBlock W plates span [z_pix[p0] - 130.005, z_pix[p0] - 15.005] mm;
    //   anchor near cal_start if consistent, else z_pix[p0] - 25.0 mm.
    const double z_first_plane = z_cl[plane_cls[p0][0]];
    double prior_z = z_first_plane - 17.505;
    if (p0 == 0) {
        prior_z = z_first_plane + 7.505;
    } else if (p0 >= 6) {
        prior_z = (has_cal && z_cal < z_first_plane && z_cal >= z_first_plane - 135.0)
            ? z_cal
            : (z_first_plane - 25.0);
    }
    out_feat[2] = p0_mip_frac;

    if (cands.empty()) {
        out_pv_xyz[0] = xc_win;
        out_pv_xyz[1] = yc_win;
        out_pv_xyz[2] = prior_z;
        return;
    }

    // Deterministic reflection-symmetric sorting: (n_hits desc, chi2 asc, slope^2 asc)
    std::stable_sort(cands.begin(), cands.end(), [](const CandTrk& a, const CandTrk& b) {
        if (a.trk.n_hits != b.trk.n_hits) return a.trk.n_hits > b.trk.n_hits;
        if (std::abs(a.trk.chi2 - b.trk.chi2) > 1e-9) return a.trk.chi2 < b.trk.chi2;
        const double sa = a.trk.tx * a.trk.tx + a.trk.ty * a.trk.ty;
        const double sb = b.trk.tx * b.trk.tx + b.trk.ty * b.trk.ty;
        return sa < sb;
    });

    std::vector<uint8_t> used_cl(n_cl, 0);
    std::vector<Tracklet3D> trks;
    trks.reserve(32);
    int32_t n_3hit = 0;

    for (const auto& c : cands) {
        if (used_cl[c.c0] || used_cl[c.c1] || (c.c2 >= 0 && used_cl[c.c2])) continue;
        used_cl[c.c0] = 1;
        used_cl[c.c1] = 1;
        if (c.c2 >= 0) used_cl[c.c2] = 1;
        trks.push_back(c.trk);
        if (c.trk.n_hits == 3) ++n_3hit;
        if (trks.size() >= 32) break;
    }

    const int32_t n_trk = static_cast<int32_t>(trks.size());
    out_feat[0] = static_cast<float>(std::log1p(static_cast<double>(n_trk)) / 3.5);
    out_feat[1] = static_cast<float>(std::log1p(static_cast<double>(n_3hit)) / 3.0);

    // 3. Adaptive Primary Vertex (AVR) Fit if >= 2 tracklets exist
    double v_pv[3] = {xc_win, yc_win, prior_z};

    if (n_trk >= 2) {
        std::vector<double> base_w(n_trk);
        for (int32_t i = 0; i < n_trk; ++i) {
            base_w[i] = (trks[i].n_hits == 3 ? 1.0 : 0.35) / std::max(trks[i].sigma2, 1e-6);
        }
        std::vector<double> cur_w = base_w;

        // Weak physical slab prior (sigma_z = 18 mm) regularizes collinear bundles without pulling exact 3D stereo intersections
        const double prior_xy_prec = 1.0 / (35.0 * 35.0);
        const double prior_z_prec = 1.0 / (18.0 * 18.0);
        const double z_clamp_lo = (p0 == 0) ? (z_first_plane - 10.0) : ((p0 >= 6) ? (z_first_plane - 135.0) : (z_first_plane - 30.0));
        const double z_clamp_hi = (p0 == 0) ? (z_first_plane + 22.0) : (z_first_plane + 18.0);

        bool ok = false;
        const double temps[3] = {4.0, 2.0, 1.0};
        for (int iter = 0; iter < 3; ++iter) {
            double v_try[3];
            if (!fit_weighted_lines_vertex(trks, cur_w, xc_win, yc_win, prior_z, prior_xy_prec, prior_z_prec, v_try)) {
                break;
            }
            v_try[2] = std::clamp(v_try[2], z_clamp_lo, z_clamp_hi);
            v_pv[0] = v_try[0];
            v_pv[1] = v_try[1];
            v_pv[2] = v_try[2];
            ok = true;

            const double T = temps[iter];
            for (int32_t i = 0; i < n_trk; ++i) {
                const double dz = std::max(1.0, std::abs(trks[i].z0 - v_pv[2]));
                const double sig2_i = trks[i].sigma2 + (kThetaMS * dz) * (kThetaMS * dz);
                const double dx = (trks[i].x0 + trks[i].tx * (v_pv[2] - trks[i].z0)) - v_pv[0];
                const double dy = (trks[i].y0 + trks[i].ty * (v_pv[2] - trks[i].z0)) - v_pv[1];
                const double chi2_i = (dx * dx + dy * dy) / sig2_i;
                const double fermi = 1.0 / (1.0 + std::exp(std::clamp((chi2_i - 9.0) / (2.0 * T), -20.0, 20.0)));
                cur_w[i] = base_w[i] * fermi;
            }
        }

        if (ok) {
            *out_pv_valid = 1;
            out_pv_xyz[0] = v_pv[0];
            out_pv_xyz[1] = v_pv[1];
            out_pv_xyz[2] = v_pv[2];

            // 4. Compute per-tracklet Impact Parameters d0, significances d0/sigma, and thrust-signed d0
            // restricted to physical tau/charm micron-to-millimeter weak-decay band (d0 <= 1.8 mm, rejecting wide gamma->ee conversions).
            double sum_tx = 0.0, sum_ty = 0.0;
            for (const auto& t : trks) {
                sum_tx += t.tx;
                sum_ty += t.ty;
            }
            const double thrust_norm = std::hypot(sum_tx, sum_ty);
            const double ux = (thrust_norm > 1e-6) ? (sum_tx / thrust_norm) : 0.0;
            const double uy = (thrust_norm > 1e-6) ? (sum_ty / thrust_norm) : 0.0;

            std::vector<double> d0_mm;
            std::vector<double> ip_sig;
            d0_mm.reserve(n_trk);
            ip_sig.reserve(n_trk);
            double max_signed_d0 = 0.0;
            double chi2_1v_sum = 0.0;
            int32_t n_displaced = 0;

            for (int32_t i = 0; i < n_trk; ++i) {
                const double dz = std::max(1.0, std::abs(trks[i].z0 - v_pv[2]));
                const double sig_i = std::sqrt(trks[i].sigma2 + (kThetaMS * dz) * (kThetaMS * dz));
                const double dx = (trks[i].x0 + trks[i].tx * (v_pv[2] - trks[i].z0)) - v_pv[0];
                const double dy = (trks[i].y0 + trks[i].ty * (v_pv[2] - trks[i].z0)) - v_pv[1];
                const double d0 = std::hypot(dx, dy);
                const double sig_val = d0 / std::max(sig_i, 0.005);
                chi2_1v_sum += std::min(sig_val * sig_val, 25.0);
                if (d0 <= 1.80) {
                    d0_mm.push_back(d0);
                    ip_sig.push_back(sig_val);
                    if (d0 >= 0.015 && sig_val > 2.5) ++n_displaced;
                    if (thrust_norm > 1e-6) {
                        const double signed_proj = -(dx * ux + dy * uy);
                        if (signed_proj > max_signed_d0) max_signed_d0 = signed_proj;
                    }
                } else {
                    d0_mm.push_back(0.0);
                    ip_sig.push_back(0.0);
                }
            }

            const double pv_chi2_ndf = chi2_1v_sum / std::max(1, 2 * n_trk - 3);
            out_feat[3] = static_cast<float>(std::asinh(pv_chi2_ndf) / 3.0);

            std::vector<double> d0_sorted = d0_mm;
            std::vector<double> sig_sorted = ip_sig;
            std::sort(d0_sorted.begin(), d0_sorted.end(), std::greater<double>());
            std::sort(sig_sorted.begin(), sig_sorted.end(), std::greater<double>());

            out_feat[4] = static_cast<float>(std::asinh(d0_sorted[0] / 0.05) / 3.0);
            out_feat[5] = static_cast<float>(std::asinh((n_trk >= 2 ? d0_sorted[1] : 0.0) / 0.05) / 3.0);
            out_feat[6] = static_cast<float>(std::asinh(sig_sorted[0] / 3.0) / 2.5);
            out_feat[7] = static_cast<float>(std::asinh((n_trk >= 2 ? sig_sorted[1] : 0.0) / 3.0) / 2.5);
            out_feat[8] = static_cast<float>(std::log1p(static_cast<double>(n_displaced)) / 2.5);
            out_feat[9] = static_cast<float>(static_cast<double>(n_displaced) / static_cast<double>(n_trk));
            out_feat[10] = static_cast<float>(std::asinh(max_signed_d0 / 0.05) / 3.0);

            // 5. Candidate Displaced Secondary Vertex Fit (2V vs 1V Delta-Chi2)
            if (n_trk >= 3 && sig_sorted[0] > 2.0) {
                std::vector<int32_t> ord_sig(n_trk);
                std::iota(ord_sig.begin(), ord_sig.end(), 0);
                std::stable_sort(ord_sig.begin(), ord_sig.end(), [&](int32_t a, int32_t b) {
                    return ip_sig[a] > ip_sig[b];
                });

                std::vector<Tracklet3D> sv_trks;
                std::vector<double> sv_w;
                for (int32_t idx : ord_sig) {
                    if (ip_sig[idx] > 2.0 || sv_trks.size() < 2) {
                        sv_trks.push_back(trks[idx]);
                        sv_w.push_back(1.0 / std::max(trks[idx].sigma2, 1e-6));
                        if (sv_trks.size() >= 6 && ip_sig[idx] <= 2.0) break;
                    }
                }
                double v_sv[3];
                if (sv_trks.size() >= 2 &&
                    fit_weighted_lines_vertex(
                        sv_trks, sv_w, v_pv[0], v_pv[1], v_pv[2] + 2.0,
                        1.0 / (20.0 * 20.0), 1.0 / (15.0 * 15.0), v_sv)) {
                    v_sv[2] = std::clamp(v_sv[2], v_pv[2], z_first_plane + 35.0);
                    double chi2_sv_old = 0.0, chi2_sv_new = 0.0;
                    for (const auto& st : sv_trks) {
                        const double dz1 = std::max(1.0, std::abs(st.z0 - v_pv[2]));
                        const double s1 = st.sigma2 + (kThetaMS * dz1) * (kThetaMS * dz1);
                        const double dx1 = (st.x0 + st.tx * (v_pv[2] - st.z0)) - v_pv[0];
                        const double dy1 = (st.y0 + st.ty * (v_pv[2] - st.z0)) - v_pv[1];
                        chi2_sv_old += (dx1 * dx1 + dy1 * dy1) / s1;

                        const double dz2 = std::max(1.0, std::abs(st.z0 - v_sv[2]));
                        const double s2 = st.sigma2 + (kThetaMS * dz2) * (kThetaMS * dz2);
                        const double dx2 = (st.x0 + st.tx * (v_sv[2] - st.z0)) - v_sv[0];
                        const double dy2 = (st.y0 + st.ty * (v_sv[2] - st.z0)) - v_sv[1];
                        chi2_sv_new += (dx2 * dx2 + dy2 * dy2) / s2;
                    }
                    const double delta_chi2 = std::max(0.0, chi2_sv_old - chi2_sv_new);
                    const double lxy = std::hypot(v_sv[0] - v_pv[0], v_sv[1] - v_pv[1]);
                    const double lz = std::max(0.0, v_sv[2] - v_pv[2]);
                    if (delta_chi2 > 3.0 && lxy <= 1.80) {
                        out_feat[11] = 1.0f;
                        out_feat[12] = static_cast<float>(std::log1p(delta_chi2) / 4.0);
                        out_feat[13] = static_cast<float>(std::asinh(lxy / 0.10) / 3.0);
                        out_feat[14] = static_cast<float>(std::asinh(lz / 2.0) / 3.0);
                    }
                }
            }
        }
    } else if (n_trk == 1) {
        const double dz_back = (p0 == 0) ? -7.505 : ((p0 >= 6) ? (z_first_plane - prior_z) : 17.505);
        out_pv_xyz[0] = trks[0].x0 - trks[0].tx * dz_back;
        out_pv_xyz[1] = trks[0].y0 - trks[0].ty * dz_back;
        out_pv_xyz[2] = trks[0].z0 - dz_back;
        *out_pv_valid = 1;
    }

    // 6. Explicit 1-Prong Parent -> Daughter Kink Angle Search across upstream planes
    double max_kink_mrad = 0.0;
    for (int32_t i = 0; i < n_trk; ++i) {
        const auto& td = trks[i];
        if (td.first_plane <= p0) continue;
        // Check clusters on upstream planes p_u < td.first_plane that lie close to td extrapolated to z_u
        // while forming a significantly kinked angle relative to out_pv_xyz -> cl_u
        for (int32_t pu = p0; pu < td.first_plane; ++pu) {
            for (int32_t cu : plane_cls[pu]) {
                const double zu = z_cl[cu];
                const double dz_kink = td.z0 - zu;
                if (dz_kink < 5.0 || dz_kink > 95.0) continue;
                // Check if kink occurred between zu and td.z0: line from PV through cu vs downstream tracklet td
                const double dz_pv = std::max(3.0, zu - out_pv_xyz[2]);
                const double tx_up = (x_mean[cu] - out_pv_xyz[0]) / dz_pv;
                const double ty_up = (y_mean[cu] - out_pv_xyz[1]) / dz_pv;
                if (std::hypot(tx_up, ty_up) > kMaxSlope) continue;

                // Midpoint intersection compatibility between upstream stub and downstream tracklet
                const double z_mid_k = 0.5 * (zu + td.z0);
                const double x_up_m = x_mean[cu] + tx_up * (z_mid_k - zu);
                const double y_up_m = y_mean[cu] + ty_up * (z_mid_k - zu);
                const double x_dn_m = td.x0 + td.tx * (z_mid_k - td.z0);
                const double y_dn_m = td.y0 + td.ty * (z_mid_k - td.z0);
                const double miss_mm = std::hypot(x_up_m - x_dn_m, y_up_m - y_dn_m);
                if (miss_mm <= 0.12) {
                    const double kink_rad = std::hypot(td.tx - tx_up, td.ty - ty_up);
                    const double kink_mrad = 1000.0 * kink_rad;
                    if (kink_mrad > max_kink_mrad) {
                        max_kink_mrad = kink_mrad;
                    }
                }
            }
        }
    }
    out_feat[15] = static_cast<float>(std::asinh(max_kink_mrad / 5.0) / 3.0);
}

}  // extern "C"
