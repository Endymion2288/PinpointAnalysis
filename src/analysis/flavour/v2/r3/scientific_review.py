"""Apply the frozen v2 practical-effect and quality limits to paired R3 results."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json


def promotion_checks(metrics):
    def item(name):
        value=metrics.get(name,{})
        point=value.get('difference_a_minus_b')
        bounds=value.get('95pct',[None,None])
        valid=point is not None and len(bounds)==2 and all(v is not None for v in bounds)
        if not valid or not np.isfinite([point,*bounds]).all():
            return None
        return point,bounds[0],bounds[1]

    ba,nll,tau,nll4=(item(k) for k in ['BA3','NLL3','tau_eff_1pct','NLL4'])
    common={
        'BA3_point_at_least_2pp':ba is not None and ba[0]>=.020,
        'BA3_positive_95pct_lower':ba is not None and ba[1]>0,
        'NLL3_increase_95pct_upper_at_most_0p010':nll is not None and nll[2]<=.010,
    }
    for key in ['recall_e','recall_mu','recall_NC']:
        recall=item(key)
        common[key+'_loss_95pct_upper_at_most_1pp']=recall is not None and recall[1]>=-.010
    tau_checks={
        'tau_eff_1pct_point_at_least_1pp':tau is not None and tau[0]>=.010,
        'tau_eff_1pct_positive_95pct_lower':tau is not None and tau[1]>0,
        'BA3_loss_95pct_upper_at_most_0p5pp':ba is not None and ba[1]>=-.005,
        'NLL4_increase_95pct_upper_at_most_0p010':nll4 is not None and nll4[2]<=.010,
    }
    return {'common_three_class':{'passed':all(common.values()),'checks':common,
                'failed':[k for k,v in common.items() if not v]},
            'tau':{'passed':all(tau_checks.values()),'checks':tau_checks,
                'failed':[k for k,v in tau_checks.items() if not v]}}


def review(paths):
    contrasts=[]
    for path in paths:
        result=json.loads(path.read_text())
        if result['status']!='complete' or result['replicates']!=400:
            raise ValueError('scientific review requires completed 400-draw evaluations')
        for name in ['independent_points.json','independent_core_intervals.json']:
            independent=json.loads((path.parent/name).read_text())
            if independent['status']!='complete' or independent['failed']:
                raise ValueError('independent point/interval review must pass')
        for pair in result['pairs']:
            if pair['domain']!='all':continue
            contrasts.append({**{k:pair[k] for k in ['a','b','reason','domain']},
                'evaluation':str(path),'primary_endpoint':pair['a'].endswith('_endpoint'),
                'metrics':pair['metrics'],'promotion':promotion_checks(pair['metrics'])})
    return {'status':'complete','contrasts':contrasts,
        'policy':'unchanged roadmap/v2/02_tasks_and_evaluation.md section6; differences candidate minus reference',
        'interpretation':'practical-effect and quality eligibility on repeated development Tune; matched seed/resolution and mechanism controls remain conditional; not independent confirmation',
        'Cal_A_Cut_B_Confirm_payload_read':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.evaluations)
    write_json(args.output,result)
    print(json.dumps({'status':result['status'],'contrasts':len(result['contrasts']),
        'eligible_primary':[r['a'] for r in result['contrasts'] if r['primary_endpoint'] and
            (r['promotion']['common_three_class']['passed'] or r['promotion']['tau']['passed'])]}),flush=True)


if __name__=='__main__':main()
