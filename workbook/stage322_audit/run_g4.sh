#!/usr/bin/env bash
set -eo pipefail
audit_dir=$(realpath "$1")
g4=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/g4
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh
export PINPOINT_SKIP_GDML=1
export PYTHIA8DATA=/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/share/Pythia8/xmldoc
probe_dir=$(mktemp -d /tmp/pinpoint322.XXXXXX)
trap 'rm -rf "$probe_dir"' EXIT
cd "$probe_dir"
pids=()
for x in 0 80; do
    for particle in mu- mu+; do
        mag=0.57
        seed=$((322100 + x))
        if [[ $particle == mu+ ]]; then seed=$((seed + 11)); fi
        name="probe_${x}_${particle}"
        cat > "$name.mac" <<MAC
/control/execute $g4/macros/geom.mac
/det/enableFaserSpectrometer true
/gen/select gun
/gps/particle $particle
/gps/pos/type Point
/gps/pos/centre $x -10 -3010 mm
/gps/direction 0 0 1
/gps/ene/type Mono
/gps/ene/mono 50 GeV
/out/fileName $audit_dir/$name.root
/out/saveTrack false
/out/saveTruthHits true
/random/setSeeds $seed $((seed + 17))
/run/initialize
/Pinpoint/field/value $mag tesla
/run/beamOn 12
MAC
        "$g4/pinpoint_fixed" "$name.mac" > "$audit_dir/$name.log" 2>&1 &
        pids+=("$!")
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
