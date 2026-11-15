"""Compact occupied sensor rows with explicit physical spacing and sensor type."""

import numpy as np
import torch
from torch import nn

from analysis.flavour.v2.r1.data import image_geometry, projection


def sensor_row_geometry(geometry):
    physical=image_geometry(geometry)
    pixel=np.asarray(geometry['pixel_Zpos'])
    scint=np.asarray(geometry['scint_Zpos'])
    rows=np.unique(np.r_[pixel,scint])
    physical_rows=np.floor(rows-physical['origin_mm'][2]).astype('int64')
    if len(np.unique(physical_rows))!=len(rows):
        raise ValueError('sensor rows require a separate response mapping for coincident 1mm bins')
    return {**physical,'height':len(rows),'physical_grid':physical,
        'row_z_mm':rows.tolist(),'physical_row':physical_rows.tolist(),
        'row_is_pixel':np.isin(rows,pixel).astype(int).tolist(),
        'row_is_bar':np.isin(rows,scint).astype(int).tolist(),
        'coordinate_channels':['z_over3000mm','log1p_previous_gap_mm_over5',
            'log1p_next_gap_mm_over5','pixel_plane','bar_plane'],
        'z_layout':'sorted actual sensor centers; row ordinal is not a physical distance',
        'bar_projection':physical['bar_projection']}


def sensor_projection(observation,geometry,image):
    # Only remove known empty z rows from the accepted projection. Preserve
    # every horizontal bin, view, response, duplicate and accumulation order.
    physical=image['physical_grid']
    indexes,values=projection(observation,geometry,physical)
    h,w=physical['height'],physical['width']
    channel=indexes//(h*w)
    old_row=(indexes//w)%h
    mapping=np.full(h,-1,dtype='int64')
    mapping[np.asarray(image['physical_row'])]=np.arange(image['height'])
    row=mapping[old_row]
    if np.any(row<0):raise ValueError('response is outside the known sensor rows')
    return (channel*image['height']+row)*w+indexes%w,values


class SensorRowEncoder(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.image=base.image
        self.stem,self.layers,self.norm=base.stem,base.layers,base.norm
        self.output_dim=base.output_dim
        conv=self.stem[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261122)
            expanded=nn.Conv2d(9,conv.out_channels,conv.kernel_size,conv.stride,conv.padding,bias=False)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:,:4].copy_(conv.weight)
        self.stem[0]=expanded
        z=np.asarray(self.image['row_z_mm'])
        gap=np.diff(z)
        coordinates=np.stack([z/3000.,np.log1p(np.r_[0.,gap])/5.,
            np.log1p(np.r_[gap,0.])/5.,self.image['row_is_pixel'],self.image['row_is_bar']])
        self.register_buffer('sensor_coordinates',torch.from_numpy(coordinates.astype('float32'))[None,:,:,None])

    def forward(self,batch):
        b=len(batch['label']);h,w=self.image['height'],self.image['width']
        dense=batch['projection_values'].new_zeros(b*8*h*w)
        dense.index_add_(0,batch['projection_event']*(8*h*w)+batch['projection_indices'],batch['projection_values'])
        dense=dense.reshape(b*2,4,h,w)
        active=dense.flatten(1).any(1)
        totals=torch.log1p(dense.sum((-1,-2))).reshape(b,8)
        coordinates=self.sensor_coordinates.expand(b*2,-1,-1,w)
        x=self.layers(self.stem(torch.cat([torch.log1p(dense),coordinates],1)))
        pooled=torch.cat([x.mean((-1,-2)),x.amax((-1,-2))],-1)
        pooled=(pooled*active[:,None]).reshape(b,-1)
        return self.norm(torch.cat([pooled.float(),totals],-1))
