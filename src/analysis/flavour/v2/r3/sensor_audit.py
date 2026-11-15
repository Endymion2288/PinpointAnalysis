"""Check exact measurement conservation when removing empty projection rows."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.data import projection
from analysis.flavour.v2.r3.data import R3Dataset
from analysis.flavour.v2.r3.sensor_rows import sensor_projection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    campaign=json.loads((args.root/'campaign.json').read_text())
    data=R3Dataset(Path(campaign['r0'])/'profile_data','train',
        {**config,'cache_training_inputs':False})
    # Include both occupancy tails plus a fixed random sample, using Train only.
    occupancy=data.metadata[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).to_numpy()
    extremes=np.r_[np.argsort(occupancy,kind='stable')[:32],np.argsort(occupancy,kind='stable')[-32:]]
    remaining=np.setdiff1d(np.arange(len(data)),extremes)
    selected=np.r_[extremes,np.random.default_rng(20261122).choice(remaining,192,replace=False)]
    image=config['image'];w=image['width'];h=image['height'];old_h=image['physical_grid']['height']
    channels=np.zeros(8,dtype='float64');measurements=0
    for index in selected:
        observation=data.native[int(data.indexes[index])]['observations']
        original,values=projection(observation,data.geometry,image['physical_grid'])
        compact,new_values=sensor_projection(observation,data.geometry,image)
        restored=((compact//(h*w))*old_h+np.asarray(image['physical_row'])[(compact//w)%h])*w+compact%w
        if not np.array_equal(original,restored) or not np.array_equal(values,new_values):
            raise ValueError('compact sensor rows changed accepted measurements')
        if len(compact) and not (compact.min()>=0 and compact.max()<8*h*w):
            raise ValueError('compact sensor projection index outside image')
        measurements+=len(values)
        channels+=np.bincount(compact//(h*w),weights=values,minlength=8)
    result={'status':'complete','config':config,'Train_events':len(selected),
        'class_counts':np.bincount(data.metadata.iloc[selected].label,minlength=4).tolist(),
        'index_selection':'lowest32/highest32 occupancy plus192 fixed random remaining Train events',
        'all_measurement_indices_and_values_exactly_reversible':True,
        'response_measurements':measurements,'view_channel_sums':channels.tolist(),
        'physical_shape':[old_h,w],'compact_shape':[h,w],
        'all_physical_z_and_gaps_explicit':True,
        'row_z_mm':image['row_z_mm'],'gap_quantiles_mm':np.quantile(np.diff(image['row_z_mm']),[0,.5,1]).tolist(),
        'bar_unmeasured_coordinate_policy':image['bar_projection'],
        'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False}
    path=args.root/'audit'/f'sensor_rows_{args.attempt}.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,result)
    data.native.close()
    print(json.dumps({k:result[k] for k in ['status','Train_events','response_measurements','physical_shape','compact_shape']}),flush=True)


if __name__=='__main__':main()
