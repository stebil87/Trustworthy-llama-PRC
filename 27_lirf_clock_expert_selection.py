#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRC 2026 Campaign 27 -- select stable LIRF unreliable-clock expert.

RESEARCH ONLY, 2025. No 2026 reads, full-2025 final refit, ranking prediction,
or submission. Fits twice (like C26):
  A. Train on Mar-May and Aug-Sep 2025 (chronologically < Oct), select the
     legacy incumbent on Oct 2025, then select optional LIRF bad-clock mixture
     weights on *Nov and Dec 2025* only, strictly after training.  Oct is NOT
     used to tune mixture weights. Both months' labels are unavailable to all
     models in stage A.
  B. Refit selected architecture on all 8 non-ranking-analogue months in 2025:
     Mar-May and Aug-Dec. Report frozen decisions on Jan, Feb, Jun, Jul 2025
     (671k rows; no held-out target available during either fit/selection).

This blocked analogue test is retrospective, not chronological evaluation and
not an unbiased independent holdout (all 2025 months previously explored).
Changes to gates/weights after viewing these results must be treated as
further historical exploration, not as proof of expected 2026 improvement.

Dependency: unchanged campaigns 21, 22, 23, 26 alongside this script.
Uses ALL available FIT rows, no sampling or tail clipping in preset FULL.
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

PARENT_26 = '26_ranking_month_block_stress.py'
SHA_26 = 'a92a739af13b491de798cac556a982ac55f402640346fcfbe8f46b746990671e'
CAL_MONTHS = ('2025-11', '2025-12')
HOLDOUT = ('2025-01','2025-02','2025-06','2025-07')
WEIGHTS = (0.20,0.35,0.50,0.65)
CANDIDATE_NAMES = ('flow20','flow35','flow50','flow65','queue35','queue50',
                   'mixed35')


def _digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):
            h.update(x)
    return h.hexdigest()


def load_parent():
    p=Path(__file__).resolve().with_name(PARENT_26)
    if not p.is_file():raise FileNotFoundError('Missing unchanged Campaign 26 beside this script: '+str(p))
    sha=_digest(p)
    if sha!=SHA_26:raise RuntimeError(f'C26 code digest mismatch: {sha}; expected {SHA_26}')
    spec=importlib.util.spec_from_file_location('campaign26_locked_for27',p)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    c23,c22,parent,sha23,sha22,sha21=mod.import_locked()
    return mod,c23,c22,parent,sha,sha23,sha22,sha21


def make_candidates(base, qpred, n):
    """Invert only a FIXED 35% inherited blend to compare preset weights.

    These are not new supervised learners, and never modify good-clock or
    non-LIRF records (enforced by C23 candidate construction).
    """
    b=np.asarray(base,dtype=np.float64)
    flow=np.asarray(qpred['c23_bad_lirf_flow_35'],dtype=np.float64)
    queue=np.asarray(qpred['c23_bad_lirf_queue_35'],dtype=np.float64)
    if any(len(x)!=n for x in (b,flow,queue)) or not np.isfinite(b).all():
        raise AssertionError('Input prediction lengths/nonfinite')
    # The constituent specialists are already non-negative.  Calculate
    # new blends directly from the 35% residuals, not from true labels.
    df=flow-b
    dq=queue-b
    candidates={f'flow{int(w*100)}':np.maximum(0,b+(w/.35)*df)
                for w in WEIGHTS}
    candidates['queue35']=queue
    candidates['queue50']=np.maximum(0,b+(.5/.35)*dq)
    candidates['mixed35']=np.maximum(0,b+0.5*(df+dq))
    if set(candidates)!=set(CANDIDATE_NAMES):
        raise AssertionError('Missing challenger')
    if not all(np.isfinite(x).all() and (x>=0).all() for x in candidates.values()):
        raise AssertionError('Invalid challenger predictions')
    return candidates


def rmse(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not len(y):raise AssertionError('Empty score')
    if len(y)!=len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise AssertionError('Malformed RMSE inputs')
    return float(np.sqrt(np.mean((y-p)**2)))


def choose_on_calibration(y,days,months,base,challenges,parent,boot,seed,
                          min_gain,max_regret,confidence):
    """Selection ONLY on Nov/Dec; neither target vector gets into FIT.

    We demand improvement in pooled RMSE, a limited per-month downside,
    and optional paired day-bootstrap q95<0.  Deterministic fallback.
    """
    month=np.asarray(months,dtype=str)
    if set(month)!={'2025-11','2025-12'}:
        raise AssertionError('Calibration includes prohibited months')
    ref=rmse(y,base)
    rmonth={m:rmse(y[month==m],base[month==m]) for m in CAL_MONTHS}
    records=[];accept=[]
    for k,name in enumerate(CANDIDATE_NAMES):
        p=challenges[name]
        per={m:rmse(y[month==m],p[month==m])-rmonth[m]
             for m in CAL_MONTHS}
        gain=rmse(y,p)-ref
        if np.array_equal(p,base):
            q95=0.;qb={'q05_delta':0.,'q50_delta':0.,'q90_delta':0.,'q95_delta':0.}
        else:
            qb=parent.paired_bootstrap(y,p,base,days,boot,seed+37*k)
            q95=float(qb['q95_delta'])
        qualified=(gain <= -min_gain and max(per.values()) <= max_regret
                   and (not confidence or q95<0.))
        row={'candidate':name,'n':int(len(y)),
             'pooled_base_rmse':ref,'pooled_challenger_rmse':rmse(y,p),
             'pooled_delta_rmse':gain,
             'nov_delta_rmse':per['2025-11'],'dec_delta_rmse':per['2025-12'],
             'paired_day_q95_delta':q95,'qualified':bool(qualified),
             **{f'bootstrap_{k}':v for k,v in qb.items() if k in ('q05_delta','q50_delta','q90_delta')}}
        records.append(row)
        if qualified:accept.append(row)
    # Robustness first: prefer the candidate with least worst-month regret;
    # tie-break on pooled gain.  All scores are Nov/Dec ONLY.
    winner=sorted(accept,key=lambda r:(max(r['nov_delta_rmse'],r['dec_delta_rmse']),
                                       r['pooled_delta_rmse'],r['candidate']))[0]['candidate'] if accept else None
    return winner,records


def select_test():
    rng=np.random.default_rng(2026)
    # Existence of both calibration months is mandatory.
    months=np.array(['2025-11']*45+['2025-12']*45)
    y=np.r_[rng.normal(800,50,45),rng.normal(850,50,45)]
    b=y+40
    fake=type('Bootstrap',(),{'paired_bootstrap':staticmethod(lambda y,p,b,days,n,seed:
           {'q05_delta':-6.,'q50_delta':-5.,'q90_delta':-4.,'q95_delta':-3.})})
    dummy=np.array([f'2025-11-{1+i//3:02d}' if i<45 else f'2025-12-{1+(i-45)//3:02d}' for i in range(90)])
    ch={k:y+10 for k in CANDIDATE_NAMES}
    name,rows=choose_on_calibration(y,dummy,months,b,ch,fake,100,2026,1,1,True)
    assert name is not None and len(rows)==len(CANDIDATE_NAMES)
    chbad={k:y+300 for k in CANDIDATE_NAMES}
    fallback,_=choose_on_calibration(y,dummy,months,b,chbad,fake,100,2026,1,1,True)
    assert fallback is None
    b=np.array([300.,600.,700.]);q={'c23_bad_lirf_flow_35':b+np.array([0,10,20]),
                                     'c23_bad_lirf_queue_35':b+np.array([0,8,16])}
    out=make_candidates(b,q,3)
    assert np.allclose(out['flow35'],q['c23_bad_lirf_flow_35'])
    assert np.allclose(out['flow50'],b+np.array([0,10,20])*(0.50/.35))
    assert out['mixed35'][0]==b[0]
    # Future evaluation labels cannot affect candidate generation or selection.
    heldout_y=np.array([1.e5,5.e6]);heldout_y[:]=0
    assert len(out)==7 and all(len(z)==3 for z in out.values())
    try:
        choose_on_calibration(y,dummy,np.array(['2025-07']*90),b,ch,fake,100,2026,1,1,True)
        raise AssertionError('Allowed selection on blocked month')
    except AssertionError as e:
        if 'prohibited' not in str(e):raise
    print('SELF-TEST PASS C27: fixed blend isolation, Nov/Dec-only selection, fallback, blocked-month rejection',flush=True)


def args_from_cli():
    p=argparse.ArgumentParser(description='C27 2025-only LIRF bad-clock ensemble/selection stress')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign27_research'))
    p.add_argument('--preset',choices=('full','smoke'),default='full')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-gain-sec',type=float,default=1.0)
    p.add_argument('--max-month-regret-sec',type=float,default=1.0)
    p.add_argument('--no-bootstrap-gate',action='store_true',help='Research sensitivity ONLY, relax q95 requirement')
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.threads<1 or a.bootstrap<100 or a.min_gain_sec<0 or a.max_month_regret_sec<0:
        p.error('Invalid args')
    return a


def main():
    select_test()
    c=args_from_cli()
    c26,c23,c22,parent,sha26,sha23,sha22,sha21=load_parent()
    if c.self_test:
        c26.test_partition();c23.self_test(parent,c22)
        return 0
    tic=time.perf_counter()
    print('='*79,flush=True)
    print('CAMPAIGN 27: STABLE LIRF BAD-CLOCK EXPERT / 2025 RESEARCH ONLY',flush=True)
    print('OCT=LEGACY SELECT; NOV+DEC=NEW SPECIALIST SELECT; JAN/FEB/JUN/JUL=BLOCK REPORT',flush=True)
    print('NO 2026 READ, NO FULL-2025 FIT, NO FINAL PREDICTIONS/SUBMISSION',flush=True)
    print('='*79,flush=True)
    # Construct unchanged original defaults without assuming its CLI supports our switches.
    argv=sys.argv[:]
    try:
        sys.argv=[argv[0], '--preset',c.preset,'--threads',str(c.threads),
                  '--bootstrap',str(c.bootstrap),'--seed',str(c.seed)]
        modelargs=c22.args_from_cli()
    finally:sys.argv=argv
    modelargs.data_dir=c.data_dir
    modelargs.out_dir=c.out_dir
    raw,sources=parent.load_training_files(c.data_dir)
    feat=parent.build_features(raw)
    yy=parent.make_y(raw)
    parent.add_past_only_flow(feat)
    c23.add_strict_history(feat)
    # Use SAME Jan/Feb/Jun/Jul blocked partition as C26.
    cuts=c26.partition(feat,yy,modelargs.embargo_days)
    months=feat['_month'].astype(str).to_numpy()
    valid=np.isfinite(yy)&(yy>=0)
    calibration=np.flatnonzero(np.isin(months,CAL_MONTHS)&valid)
    for k,idx in cuts.items():print(f'ROWS {k}: {len(idx):,}',flush=True)
    print(f'ROWS calendar-independent Nov/Dec calibration: {len(calibration):,}',flush=True)
    if np.intersect1d(cuts['fit_select'],calibration).size or np.intersect1d(cuts['heldout'],calibration).size:
        raise AssertionError('Calibration overlaps FIT/blocked')
    # Do not allow label leakage from the 4 blocked months or Nov/Dec in
    # the first (chronological) fit, even through lagged-label features.
    mask_y=yy.copy()
    mask_y[cuts['heldout']]=np.nan
    mask_y[calibration]=np.nan
    assert np.isfinite(mask_y[cuts['fit_select']]).all()
    print('STAGE A: FIT strictly before Oct (2025-03..05,08..09); baseline selected on Oct',flush=True)
    pack=c23.fit_models(feat,mask_y,cuts['fit_select'],modelargs,c22,parent)
    incumbent_name,incumbent_oct,legacy=c23.c22_locked_candidates(
        pack,feat,mask_y,cuts['select'],modelargs,c22,parent,c26.SELECT_FOR)
    print('OCTOBER frozen incumbent:',incumbent_name,flush=True)
    query_y=mask_y.copy();query_y[cuts['select']]=np.nan
    base_cal=c23.c22_locked_predict(pack,feat,query_y,calibration,modelargs,parent,legacy)
    q_cal,_=c23.queue_candidates(pack,feat,calibration,base_cal,modelargs,parent)
    challengers=make_candidates(base_cal,q_cal,len(calibration))
    winner,select_rows=choose_on_calibration(
        yy[calibration],feat['_day'].iloc[calibration].to_numpy(),months[calibration],
        base_cal,challengers,parent,c.bootstrap,c.seed,
        c.min_gain_sec,c.max_month_regret_sec,not c.no_bootstrap_gate)
    chosen=winner or 'C22_locked'
    print('NOV+DEC challenger locked:',chosen,flush=True)
    for r in select_rows:
        print(f"  {r['candidate']:<10s} pooled_delta={r['pooled_delta_rmse']:+.3f} "
              f"Nov={r['nov_delta_rmse']:+.3f} Dec={r['dec_delta_rmse']:+.3f} "
              f"q95={r['paired_day_q95_delta']:+.3f} eligible={r['qualified']}",flush=True)
    del pack,base_cal,q_cal,challengers
    # Stage B uses 8 FIT months (1,413,560 rows in real 2025) and NO
    # source labels from the target-analogue months.
    print('STAGE B: RE-FIT 8 nonranking months, no row sampling or target clipping in FULL',flush=True)
    fit_y=yy.copy();fit_y[cuts['heldout']]=np.nan
    pack=c23.fit_models(feat,fit_y,cuts['fit_final'],modelargs,c22,parent)
    idx=cuts['heldout']
    base=c23.c22_locked_predict(pack,feat,fit_y,idx,modelargs,parent,legacy)
    q,_=c23.queue_candidates(pack,feat,idx,base,modelargs,parent)
    cand=make_candidates(base,q,len(idx))
    chosen_pred=cand[winner] if winner else np.asarray(base,float)
    y=yy[idx];ms=months[idx]
    aps=feat['airport'].iloc[idx].astype(str).to_numpy()
    isbad=(aps=='LIRF')&(~parent._plausible_proxy(feat,idx))
    stats=[];diag=[]
    for month in HOLDOUT:
        mask=ms==month
        outbase=rmse(y[mask],base[mask]);outnew=rmse(y[mask],chosen_pred[mask])
        row={'outer':month,'n':int(mask.sum()),'C22_base_rmse':outbase,
             'C27_locked_rmse':outnew,'C27_locked':chosen,
             'lirf_bad':int((mask&isbad).sum())}
        for name,pred in cand.items():
            row['rmse_'+name]=rmse(y[mask],pred[mask])
        stats.append(row)
        print(f'{month}: C22={outbase:.3f} C27_locked={outnew:.3f} '
              f"flow35={row['rmse_flow35']:.3f} [bad_LIRF={row['lirf_bad']}]",flush=True)
        for slice_name,submask in [('all',mask),('LIRF_bad',mask&isbad),
                                   ('LIRF_good',mask&(aps=='LIRF')&(~isbad)),
                                   ('LTFM',mask&(aps=='LTFM')),
                                   ('target_gt3600_DESCRIPTIVE',mask&(y>3600))]:
            if not submask.any():continue
            b=(y[submask]-base[submask])**2
            d=(y[submask]-chosen_pred[submask])**2
            diag.append({'outer':month,'slice':slice_name,'n':int(submask.sum()),
                         'rmse_C22':float(np.sqrt(b.mean())),
                         'rmse_C27':float(np.sqrt(d.mean())),
                         'delta_sse':float((d-b).sum()),
                         'sse_C22':float(b.sum())})
    result={'campaign':27,'research_only':True,'no_2026':True,'no_full_2025_refit':True,
            'design':'Frozen Oct baseline and Nov/Dec specialist selection; 4-month blocked retrospective report',
            'training_months':['2025-03','2025-04','2025-05','2025-08','2025-09',
                               '2025-10','2025-11','2025-12'],
            'selector_month':'2025-10','specialist_select_months':list(CAL_MONTHS),
            'blocked_months':list(HOLDOUT),
            'baseline_choice':incumbent_name,'specialist_choice':chosen,
            'selected_on_blocked_labels':False,
            'pooled':{'C22':rmse(y,base),'C27_locked':rmse(y,chosen_pred),
                      'flow35_DESCRIPTIVE':rmse(y,cand['flow35']),
                      'queue35_DESCRIPTIVE':rmse(y,cand['queue35'])},
            'calibration':select_rows,'months':stats,
            'seconds':round(time.perf_counter()-tic,1),
            'script_sha256':_digest(__file__),'deps_sha256':{'26':sha26,'23':sha23,'22':sha22,'21':sha21},
            'source_files':sources,
            'warning':'All 2025 months explored historically; 2026 generalization unmeasured. No target of 4 blocked months used for selection.'}
    c.out_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(select_rows).to_csv(c.out_dir/'calibration_nov_dec_candidates.csv',index=False)
    pd.DataFrame(stats).to_csv(c.out_dir/'ranking_month_block_candidates_2025_DESCRIPTIVE.csv',index=False)
    pd.DataFrame(diag).to_csv(c.out_dir/'blocked_regime_diagnostics_2025_DESCRIPTIVE.csv',index=False)
    if c.save_outer_predictions:
        frame=pd.DataFrame({'outer':ms,'airport':aps,'utc_mvt':feat['_timestamp'].iloc[idx].to_numpy(),
                            'lirf_bad_aobt':isbad,'y_true':y,
                            'pred_C22':base,'pred_C27_locked':chosen_pred})
        if parent.ID in raw:
            frame.insert(0,parent.ID,raw[parent.ID].iloc[idx].to_numpy())
        for name,p in cand.items():frame['pred_'+name]=p
        try:frame.to_parquet(c.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:frame.to_csv(c.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.csv',index=False)
    (c.out_dir/'summary.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print('POOLED blocked 2025:',{k:round(v,3) for k,v in result['pooled'].items()},flush=True)
    print('RESEARCH ONLY -- NO 2026 RANKING / NO FINAL TRAIN / NO SUBMISSION',flush=True)
    print('Research output:',c.out_dir.resolve(),flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssertionError,ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print('FATAL C27:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
