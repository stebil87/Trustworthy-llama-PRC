#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRC campaign 28: targeted weight tuning for the proven LIRF bad-clock pipeline.

RESEARCH ONLY / 2025. Inherits unchanged campaign 27 + 26 + 23 + 22 + 21.
No 2026 ranking read, no final all-2025 refit, no submission.

Stage A: Oct legacy baseline and Nov/Dec LIRF specialist selection are LOCKED
according to C27. Only if the NOV/DEC labels support an improvement versus
that C27 incumbent, test larger flow-specialist weights 75/85/95/100% with
paired-day bootstrap and Nov/Dec half-month robustness constraints.  No
blocked (Jan/Feb/Jun/Jul) target can affect either selection.
Stage B: refit same trained architecture on ALL eight non-ranking-analogue
2025 months (no downsampling in FULL), report frozen choices on the four
blocked 2025 months.  This is RETROSPECTIVE, not forward OOF; historical
2025 months have been repeatedly inspected. The block is NOT a virgin holdout.

Purpose: determine whether C27 flow65 is underweighted, not to reopen generic
OSQP, queueing or hyperparameter searches.  All non-LIRF and good-clock
predictions stay *identical* to C27's inherited baseline.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P27_NAME = '27_lirf_clock_expert_selection.py'
P27_SHA = '55df04e922072fb565c5bbcd4877180ee449e63e752474e41657ed5bdbf32a4e'
HIGH_WEIGHTS = (0.75,0.85,0.95,1.00)
BLOCK_MONTHS = ('2025-01','2025-02','2025-06','2025-07')


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_deps():
    p=Path(__file__).resolve().with_name(P27_NAME)
    if not p.is_file():
        raise FileNotFoundError('Original C27 must be next to this script: '+str(p))
    sha=digest(p)
    if sha!=P27_SHA:
        raise RuntimeError('C27 SHA-256 mismatch; original expected '+P27_SHA+', found '+sha)
    spec=importlib.util.spec_from_file_location('campaign27_locked_for28',p)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    c26,c23,c22,parent,*other=m.load_parent()
    return m,c26,c23,c22,parent


def high_candidates(base,queue_preds):
    """Reconstruct flow-expert prediction from the frozen 35%-blend.

    Never use any labels. When no expert is fitted, q_flow35 == baseline,
    therefore EVERY challenger is the unchanged baseline.
    """
    b=np.asarray(base,dtype=np.float64)
    f=np.asarray(queue_preds['c23_bad_lirf_flow_35'],dtype=np.float64)
    if b.shape != f.shape or not np.isfinite(b).all() or not np.isfinite(f).all():
        raise AssertionError('Invalid flow candidate inputs')
    return {f'flow{int(round(100*w))}':np.maximum(0.0,b+(w/0.35)*(f-b))
            for w in HIGH_WEIGHTS}


def _month_partitions(months,dates):
    """Four diagnostic contiguous half-month blocks, not random rows."""
    dates=pd.to_datetime(dates,utc=True)
    mon=np.asarray(months,dtype=str)
    day=np.asarray(dates.day,dtype=int)
    return {m+'_'+side:((mon==m)&((day<=15) if side=='first' else (day>15)))
            for m in ('2025-11','2025-12') for side in ('first','second')}


def choose_extension(y,dates,months,base27,challenges,parent,boot,seed,
                     min_gain,max_month_regret,max_half_regret,require_bootstrap):
    mon=np.asarray(months,dtype=str)
    if set(mon)!={'2025-11','2025-12'}:
        raise AssertionError('C28 extension selection must use only Nov/Dec')
    y=np.asarray(y,dtype=float); b=np.asarray(base27,dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(b).all() or len(y)!=len(b):
        raise AssertionError('Malformed calibrator input')
    halves=_month_partitions(mon,dates)
    if not all(np.count_nonzero(v)>=10 for v in halves.values()):
        raise AssertionError('Missing Nov/Dec half-month')
    ref=float(np.sqrt(np.mean((y-b)**2)))
    perref={m:float(np.sqrt(np.mean((y[mon==m]-b[mon==m])**2)))
            for m in ('2025-11','2025-12')}
    halfref={k:float(np.sqrt(np.mean((y[mask]-b[mask])**2)))
             for k,mask in halves.items()}
    records=[]; qualified=[]
    for j,(name,p0) in enumerate(challenges.items()):
        p=np.asarray(p0,dtype=float)
        if len(p)!=len(y) or not np.isfinite(p).all() or (p<0).any():
            raise AssertionError('Invalid challenger '+name)
        pooled=float(np.sqrt(np.mean((y-p)**2)))-ref
        per={m:float(np.sqrt(np.mean((y[mon==m]-p[mon==m])**2)))-perref[m]
             for m in ('2025-11','2025-12')}
        half={k:float(np.sqrt(np.mean((y[mask]-p[mask])**2)))-halfref[k]
              for k,mask in halves.items()}
        if np.array_equal(p,b):
            q95=0.0
        else:
            paired=parent.paired_bootstrap(y,p,b,np.asarray(dates),boot,seed+91*j)
            q95=float(paired['q95_delta'])
        eligible=(pooled<=-min_gain and max(per.values())<=max_month_regret
                  and max(half.values())<=max_half_regret
                  and (not require_bootstrap or q95<0))
        row={'candidate':name,'delta_pooled':pooled,'delta_nov':per['2025-11'],
             'delta_dec':per['2025-12'],'worst_half_delta':max(half.values()),
             'paired_day_q95_delta':q95,'qualified':bool(eligible),
             **{'delta_'+k:v for k,v in half.items()}}
        records.append(row)
        if eligible:qualified.append(row)
    # A challenger must beat frozen C27, not merely the weaker C22 baseline.
    # Maximize the robust worst-month improvement, then pooled gain.
    champion=min(qualified,key=lambda r:(max(r['delta_nov'],r['delta_dec']),
                                         r['delta_pooled'],r['candidate']))['candidate'] if qualified else None
    return champion,records


def self_test():
    rng=np.random.default_rng(2026)
    b=np.array([100.,1000.,2500.,8000.]); g={'c23_bad_lirf_flow_35':b+np.array([0.,70.,0.,700.])}
    hh=high_candidates(b,g)
    assert set(hh)=={'flow75','flow85','flow95','flow100'}
    assert np.array_equal(hh['flow100'][[0,2]],b[[0,2]])
    assert np.allclose(hh['flow100'],b+(g['c23_bad_lirf_flow_35']-b)/.35)
    assert all(np.array_equal(x,b) for x in high_candidates(b,{'c23_bad_lirf_flow_35':b}).values())
    months=np.array(['2025-11']*90+['2025-12']*90)
    days=np.array([f'2025-11-{j//3+1:02d}' for j in range(90)]+[f'2025-12-{j//3+1:02d}' for j in range(90)])
    truth=np.ones(180)*1000
    current=truth+np.linspace(15,30,180)
    corr={'flow75':truth+5,'flow85':truth+9,'flow95':truth+12,'flow100':truth+17}
    fake=type('Stub',(),{'paired_bootstrap':staticmethod(lambda y,p,b,d,boot,seed:
                       {'q95_delta':-1.0})})
    choice,rows=choose_extension(truth,days,months,current,corr,fake,100,2026,1,1,1,True)
    assert choice=='flow75' and len(rows)==4
    fallback,_=choose_extension(truth,days,months,current,
         {k:current+400 for k in corr},fake,100,2026,1,1,1,True)
    assert fallback is None
    try:
        choose_extension(truth,days,np.array(['2025-07']*180),current,corr,fake,100,2026,1,1,1,True)
        raise AssertionError('Target month incorrectly accepted')
    except AssertionError as e:
        if 'Nov/Dec' not in str(e):raise
    # Heldout target changes must never affect either candidate generation or
    # calibration decisions, which only accept Nov+Dec vectors.
    hlabels=rng.normal(0,1,9); hlabels[:]=1e10
    assert len(hlabels)==9 and choice=='flow75'
    print('SELF-TEST PASS C28: sparse expert fallback, original scale, month/half gating, blocked-label rejection',flush=True)


def cli():
    p=argparse.ArgumentParser(description='C28: stable high-weight LIRF flow-expert research 2025 only')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign28_research'))
    p.add_argument('--preset',choices=['full','smoke'],default='full')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-additional-gain-sec',type=float,default=1.0)
    p.add_argument('--max-month-regret-sec',type=float,default=1.0)
    p.add_argument('--max-half-regret-sec',type=float,default=3.0)
    p.add_argument('--no-bootstrap-gate',action='store_true',help='SENSITIVITY ONLY: relax confidence criterion')
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.threads<1 or a.bootstrap<100 or min(a.min_additional_gain_sec,a.max_month_regret_sec,a.max_half_regret_sec)<0:
        p.error('Invalid arguments')
    return a


def main():
    self_test()
    args=cli()
    c27,c26,c23,c22,parent=load_deps()
    if args.self_test:
        c27.select_test(); c26.test_partition(); c23.self_test(parent,c22)
        return 0
    tic=time.perf_counter()
    print('='*78,flush=True)
    print('C28 -- FLOW-EXPERT WEIGHT EXTENSION / RESEARCH 2025 ONLY',flush=True)
    print('C27 locked on Oct+Nov/Dec; C28 tunes 75/85/95/100 on Nov/Dec only',flush=True)
    print('NO 2026 READ / NO ALL-2025 FINAL FIT / NO SUBMISSION',flush=True)
    print('='*78,flush=True)
    saved=sys.argv[:]
    try:
        sys.argv=[saved[0],'--preset',args.preset,'--threads',str(args.threads),
                  '--bootstrap',str(args.bootstrap),'--seed',str(args.seed)]
        modelargs=c22.args_from_cli()
    finally:sys.argv=saved
    modelargs.data_dir=args.data_dir
    modelargs.out_dir=args.out_dir
    raw,source=parent.load_training_files(args.data_dir)
    features=parent.build_features(raw)
    target=parent.make_y(raw)
    parent.add_past_only_flow(features)
    c23.add_strict_history(features)
    cuts=c26.partition(features,target,modelargs.embargo_days)
    mm=features['_month'].astype(str).to_numpy()
    good=np.isfinite(target)&(target>=0)
    cal_idx=np.flatnonzero(np.isin(mm,c27.CAL_MONTHS)&good)
    if np.intersect1d(cal_idx,cuts['heldout']).size or np.intersect1d(cal_idx,cuts['fit_select']).size:
        raise AssertionError('Unexpected fit/calibration/blocked overlap')
    print('FIT SELECT rows',len(cuts['fit_select']),'FIT BLOCK rows',len(cuts['fit_final']),
          'NOV/DEC calibration',len(cal_idx),'BLOCK',len(cuts['heldout']),flush=True)
    # Both blocked and November/December labels suppressed in the entire
    # initial model pipeline, even through lagged-label priors.
    yy=target.copy();yy[cuts['heldout']]=np.nan;yy[cal_idx]=np.nan
    pack=c23.fit_models(features,yy,cuts['fit_select'],modelargs,c22,parent)
    name,_,legacy=c23.c22_locked_candidates(pack,features,yy,cuts['select'],
                      modelargs,c22,parent,c26.SELECT_FOR)
    basecal=c23.c22_locked_predict(pack,features,yy,cal_idx,modelargs,parent,legacy)
    qcal,_=c23.queue_candidates(pack,features,cal_idx,basecal,modelargs,parent)
    old=c27.make_candidates(basecal,qcal,len(cal_idx))
    oldwinner,oldrows=c27.choose_on_calibration(
        target[cal_idx],features['_day'].iloc[cal_idx].to_numpy(),mm[cal_idx],
        basecal,old,parent,args.bootstrap,args.seed,1.0,1.0,not args.no_bootstrap_gate)
    base27=old[oldwinner] if oldwinner else np.asarray(basecal,float)
    hi=high_candidates(basecal,qcal)
    newwinner,newrows=choose_extension(target[cal_idx],
        features['_day'].iloc[cal_idx].to_numpy(),mm[cal_idx],base27,hi,
        parent,args.bootstrap,args.seed,args.min_additional_gain_sec,
        args.max_month_regret_sec,args.max_half_regret_sec,not args.no_bootstrap_gate)
    print('Locked October baseline:',name,'C27:',oldwinner or 'C22_locked',
          'C28 additional:',newwinner or 'NONE (C27 fallback)',flush=True)
    for r in newrows:
        print(f"  {r['candidate']:<7s} dRMSE={r['delta_pooled']:+.3f} "
              f"Nov={r['delta_nov']:+.3f} Dec={r['delta_dec']:+.3f} "
              f"half_worst={r['worst_half_delta']:+.3f} q95={r['paired_day_q95_delta']:+.3f} "
              f"qualified={r['qualified']}",flush=True)
    del pack,basecal,qcal,old,hi,base27
    # ALL eight FIT months, never train on Jan/Feb/Jun/Jul target labels.
    fy=target.copy();fy[cuts['heldout']]=np.nan
    pack=c23.fit_models(features,fy,cuts['fit_final'],modelargs,c22,parent)
    idx=cuts['heldout']
    b=c23.c22_locked_predict(pack,features,fy,idx,modelargs,parent,legacy)
    q,_=c23.queue_candidates(pack,features,idx,b,modelargs,parent)
    old=c27.make_candidates(b,q,len(idx))
    cc27=old[oldwinner] if oldwinner else np.asarray(b,float)
    high=high_candidates(b,q)
    cc28=high[newwinner] if newwinner else cc27
    y=target[idx];ms=mm[idx]
    a=features['airport'].iloc[idx].astype(str).to_numpy()
    bad=(a=='LIRF') & (~parent._plausible_proxy(features,idx))
    if not np.array_equal(cc27[~bad],b[~bad]) or not np.array_equal(cc28[~bad],b[~bad]):
        raise AssertionError('Specialist modified good-clock or non-LIRF record')
    def score(m,p):return c27.rmse(y[m],p[m])
    report=[];diagnostics=[]
    masks=[('ALL',np.ones(len(y),dtype=bool)),('LIRF_BAD_AOBT',bad),
           ('LIRF_GOOD_AOBT',(a=='LIRF') & (~bad)),('LTFM',a=='LTFM'),
           ('TARGET_GT3600_DESCRIPTIVE',y>3600)]
    for m in BLOCK_MONTHS:
        sel=ms==m
        row={'outer':m,'n':int(sel.sum()),'LIRF_bad':int((sel&bad).sum()),
             'C22_RMSE':score(sel,b),'C27_RMSE':score(sel,cc27),
             'C28_RMSE':score(sel,cc28),
             **{k+'_RMSE_DESCRIPTIVE':score(sel,v) for k,v in high.items()}}
        report.append(row)
        print(f'{m}: C22={row["C22_RMSE"]:.3f} C27={row["C27_RMSE"]:.3f} '
              f'C28={row["C28_RMSE"]:.3f} bad={row["LIRF_bad"]}',flush=True)
        for label,s in masks:
            z=sel&s
            if not z.any():continue
            e=(y[z]-cc28[z])**2
            f=(y[z]-cc27[z])**2
            diagnostics.append({'outer':m,'slice':label,'n':int(z.sum()),
                                'RMSE_C27':float(np.sqrt(f.mean())),
                                'RMSE_C28':float(np.sqrt(e.mean())),
                                'delta_SSE_C28_vs_C27':float(e.sum()-f.sum())})
    pooled={'C22':c27.rmse(y,b),'C27':c27.rmse(y,cc27),'C28':c27.rmse(y,cc28)}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(oldrows).to_csv(args.out_dir/'c27_calibration_reproduced.csv',index=False)
    pd.DataFrame(newrows).to_csv(args.out_dir/'c28_nov_dec_incremental_selection.csv',index=False)
    pd.DataFrame(report).to_csv(args.out_dir/'ranking_month_block_DESCRIPTIVE.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir/'regime_diagnostics_DESCRIPTIVE.csv',index=False)
    if args.save_outer_predictions:
        result=pd.DataFrame({'outer':ms,'airport':a,'LIRF_bad_AOBT':bad,
                             'y_true':y,'pred_C22':b,'pred_C27':cc27,'pred_C28':cc28})
        if parent.ID in raw:
            result.insert(0,parent.ID,raw[parent.ID].iloc[idx].to_numpy())
        for k,p in high.items():result['pred_'+k]=p
        try:result.to_parquet(args.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:result.to_csv(args.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.csv',index=False)
    meta={'campaign':28,'research_only':True,'no_2026':True,'no_final_refit':True,
          'all_2025_source_months_ingested':True,'fit_months':list(c26.FIT_MONTHS),
          'calibration_months':list(c27.CAL_MONTHS),'blocked_months':list(BLOCK_MONTHS),
          'source_rows':len(raw),'fit_rows':len(cuts['fit_final']),
          'blocked_rows':len(idx),'baseline_oct':name,'C27_selected':oldwinner or 'C22_locked',
          'C28_additional':newwinner or 'NONE','calibration_incremental':newrows,
          'pooled_blocked_DESCRIPTIVE':pooled,'months':report,
          'code_sha256':digest(__file__),'parent_C27_sha256':P27_SHA,
          'source_files':source,'elapsed_seconds':round(time.perf_counter()-tic,2),
          'warning':'2025 historical analogue test, neither independent nor forward; no 2026 score inference'}
    (args.out_dir/'summary.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
    print('POOLED BLOCKED 2025 (DESCRIPTIVE):', {k:round(v,3) for k,v in pooled.items()},flush=True)
    print('Research outputs:',args.out_dir.resolve(),flush=True)
    print('NO 2026 READ / NO ALL-2025 FINAL FIT / NO SUBMISSION',flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssertionError,ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print('FATAL C28:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
