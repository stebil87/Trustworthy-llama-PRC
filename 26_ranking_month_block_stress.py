#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRC 2026 / Campaign 26: 2025 ranking-month BLOCKED seasonal stress.

RESEARCH ONLY. The four ranking-month analogues JAN, FEB, JUN, JUL 2025 are
excluded as one block from ALL supervised FIT and SELECT labels. A candidate
policy is selected using October 2025 labels, fitted using March-May and
August-December 2025 labels, then evaluated ONCE descriptively on the four
excluded 2025 months. This is RETROSPECTIVE blocked validation, **NOT**
forward-in-time evaluation: 2025 train labels can postdate held-out 2025 rows.
This mirrors 2026 deployment training on 2025 more closely than Jan/Feb forward
validation, but is not an independent test: all 2025 months were explored before.

Full 2025 monthly source files are ingested; no ranking/submitting/2026 files,
no final full-2025 refit, no prediction or submission for 2026.

Requires the unchanged 21, 22, 23 scripts in this directory.
OSQP is intentionally NOT added: no C24 OSQP challenge was selected in 10 folds.
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

PARENT_NAME='23_queueing_operations_research_lab.py'
PARENT_SHA='d137a102e3261109f0f59dbcf1cd211a91554337e1f180b0067ff834deb1b591'
HOLDOUT=('2025-01','2025-02','2025-06','2025-07')
SELECT_MONTH='2025-10'
SELECT_FOR='2025-11'   # The inherited selector labels the select month as OUTER-1.
FIT_MONTHS=('2025-03','2025-04','2025-05','2025-08','2025-09','2025-10','2025-11','2025-12')
BOOTSTRAP_MIN=100


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):
            h.update(buf)
    return h.hexdigest()


def import_locked():
    p=Path(__file__).resolve().with_name(PARENT_NAME)
    if not p.exists():raise FileNotFoundError('Missing locked Campaign23: '+str(p))
    actual=sha256_file(p)
    if actual!=PARENT_SHA:raise RuntimeError('Campaign23 hash mismatch: '+actual)
    spec=importlib.util.spec_from_file_location('campaign23_locked_for26',p)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    c22,parent,sha22,sha21=mod.load_deps()
    return mod,c22,parent,actual,sha22,sha21


def config_from_cli(c22):
    p=argparse.ArgumentParser(description='Campaign 26 blocked ranking-month stress, research 2025 ONLY')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign26_research'))
    p.add_argument('--preset',choices=('full','smoke'),default='full')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--embargo-days',type=int,default=1)
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--self-test',action='store_true')
    c=p.parse_args()
    if c.threads<1 or c.bootstrap<BOOTSTRAP_MIN or c.embargo_days<0:
        p.error('Invalid threads / bootstrap / embargo')
    orig=sys.argv[:]
    try:
        # Reuse unchanged C22 hyperparameters and FULL training budget.
        sys.argv=[orig[0], '--preset',c.preset,'--threads',str(c.threads),
                  '--bootstrap',str(c.bootstrap),'--seed',str(c.seed),
                  '--embargo-days',str(c.embargo_days)]
        args=c22.args_from_cli()
    finally:
        sys.argv=orig
    args.data_dir=c.data_dir
    args.out_dir=c.out_dir
    return c,args


def partition(feat,y,embargo_days):
    months=feat['_month'].astype(str).to_numpy()
    t=feat['_timestamp']
    valid=np.isfinite(y)&(y>=0)
    hold=np.flatnonzero(np.isin(months,HOLDOUT)&valid)
    fit_final=np.flatnonzero(np.isin(months,FIT_MONTHS)&valid)
    boundary=pd.Timestamp('2025-10-01',tz='UTC')-pd.Timedelta(days=embargo_days)
    fit_select=np.flatnonzero(np.isin(months,FIT_MONTHS)&(t.lt(boundary).to_numpy())&valid)
    select=np.flatnonzero((months==SELECT_MONTH)&valid)
    groups={'fit_select':fit_select,'select':select,'fit_final':fit_final,
            'heldout':hold}
    for name,a in groups.items():
        if not len(a):raise AssertionError('Empty '+name)
        if name!='heldout' and np.intersect1d(a,hold).size:
            raise AssertionError('Heldout row entered '+name)
    if np.intersect1d(fit_select,select).size:raise AssertionError('FIT/SELECT overlap')
    if np.intersect1d(select,hold).size:raise AssertionError('SELECT/heldout overlap')
    fit_months=set(months[fit_final]);select_months=set(months[select]);hold_months=set(months[hold])
    if fit_months!=set(FIT_MONTHS) or select_months!={SELECT_MONTH} or hold_months!=set(HOLDOUT):
        raise AssertionError('Month coverage mismatch')
    if not (t.iloc[fit_select].max()<t.iloc[select].min()):
        raise AssertionError('SELECT chronology broken')
    if not (set(months)==set(f'2025-{i:02d}' for i in range(1,13))):
        raise AssertionError('Not all twelve 2025 source months present')
    return groups


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not len(y):return {'n':0,'rmse':None,'mae':None}
    if len(y)!=len(p) or not np.isfinite(p).all() or (p<0).any():
        raise AssertionError('Invalid prediction vectors')
    return {'n':int(len(y)),'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'mae':float(np.mean(np.abs(y-p)))}


def diagnostics(feat,y,idx,predictions,parent,outer):
    a=feat['airport'].iloc[idx].astype(str).to_numpy()
    ok=parent._plausible_proxy(feat,idx)
    truth=y[idx]
    base=predictions['C22_locked']
    b2=(truth-base)**2
    top=b2>=np.quantile(b2,.999)
    masks={
        'ALL':np.ones(len(idx),bool),
        'LIRF_BAD_AOBT':(a=='LIRF')&(~ok),
        'LIRF_GOOD_AOBT':(a=='LIRF')&ok,
        'LIRF_TAXI_GT3600':(a=='LIRF')&(truth>3600),
        'LTFM':a=='LTFM',
        'LTFM_TAXI_GT3600':(a=='LTFM')&(truth>3600),
        'OTHER_AIRPORTS':(a!='LIRF')&(a!='LTFM'),
        'TARGET_GT3600':truth>3600,
        'C22_TOP0p1PCT_ERROR':top,
        'C22_OTHER99p9PCT':~top,
    }
    rows=[]
    for label,mask in masks.items():
        if not mask.any():continue
        row={'outer':outer,'slice':label,'n':int(mask.sum()),
             'gt3600':int((truth[mask]>3600).sum()),
             'C22_sse_share':float(b2[mask].sum()/b2.sum())}
        for name,pred in predictions.items():
            err=(truth[mask]-pred[mask])**2
            row[name+'_rmse']=float(np.sqrt(err.mean()))
            row[name+'_delta_sse_vs_C22']=float(err.sum()-b2[mask].sum())
        rows.append(row)
    return rows


def test_partition():
    z=[]
    for m in range(1,13):
        z.extend(pd.date_range(f'2025-{m:02d}-05',periods=3,freq='7D',tz='UTC'))
    f=pd.DataFrame({'_timestamp':z})
    f['_month']=f['_timestamp'].dt.strftime('%Y-%m')
    y=np.arange(len(z),dtype=float)
    cuts=partition(f,y,1)
    assert len(cuts['heldout'])==12 and len(cuts['fit_final'])==24
    assert len(cuts['select'])==3
    assert not np.intersect1d(cuts['fit_select'],cuts['heldout']).size
    assert len(cuts['fit_select'])>0 and len(cuts['fit_final'])>len(cuts['fit_select'])
    assert metrics(np.array([0,100]),np.array([10,90]))['rmse']==10.
    # Date order beyond evaluation dates is legal ONLY for blocked seasonal analogy.
    assert f['_timestamp'].iloc[cuts['fit_final']].max()>f['_timestamp'].iloc[cuts['heldout']].min()
    print('SELF-TEST PASS C26: 4-month blocked exclusion, 8-month fit, Oct SELECT, disjoint indices, no 2026',flush=True)


def main():
    test_partition()
    c23,c22,parent,sha23,sha22,sha21=import_locked()
    c,args=config_from_cli(c22)
    if c.self_test:
        c23.self_test(parent,c22)
        return 0
    tic=time.perf_counter()
    print('='*78,flush=True)
    print('CAMPAIGN 26 -- RANKING MONTH ANALOGUES / RETROSPECTIVE BLOCK STRESS',flush=True)
    print('2025-01,02,06,07 TARGETS ALL EXCLUDED FROM FIT AND SELECT',flush=True)
    print('FITTING MAY USE 2025 ROWS AFTER EVALUATED MONTHS: NOT FORWARD OOF',flush=True)
    print('NO 2026 READ / NO FINAL FULL-2025 FIT / NO SUBMISSION',flush=True)
    print('='*78,flush=True)
    raw,sources=parent.load_training_files(c.data_dir)
    f=parent.build_features(raw)
    y=parent.make_y(raw)
    parent.add_past_only_flow(f)
    c23.add_strict_history(f)
    cuts=partition(f,y,c.embargo_days)
    for name,i in cuts.items():
        print(f'{name}: {len(i):,} rows',flush=True)
    # 2025 labels from ANY of the four pseudo-ranking months are NEVER used
    # by supervised FIT or SELECT, including every specialist and tail model.
    masked_y=y.copy()
    masked_y[cuts['heldout']]=np.nan
    assert np.isfinite(masked_y[cuts['fit_select']]).all()
    assert np.isfinite(masked_y[cuts['fit_final']]).all()
    assert np.isfinite(masked_y[cuts['select']]).all()
    print('SELECT on 2025-10 with 2025-03..05 and 2025-08..09 FIT',flush=True)
    pack=c23.fit_models(f,masked_y,cuts['fit_select'],args,c22,parent)
    c22name,cpred,legacy=c23.c22_locked_candidates(
        pack,f,masked_y,cuts['select'],args,c22,parent,SELECT_FOR)
    qpred,_=c23.queue_candidates(pack,f,cuts['select'],cpred,args,parent)
    win23,selrows=c23.select_new(f,masked_y,cuts['select'],cpred,qpred,args,parent,SELECT_FOR)
    print('SELECT LOCKED: C22='+c22name+'; C23='+(win23 or 'C22')+' (OSQP omitted: 0/10 selected)',flush=True)
    # No evaluation labels are used in model choice; policy frozen here.
    del pack,cpred,qpred
    print('REFITTING on all eight non-ranking 2025 months, no row cap in FULL',flush=True)
    pack=c23.fit_models(f,masked_y,cuts['fit_final'],args,c22,parent)
    # Query all heldout records in one shot; then partition by month.
    idx=cuts['heldout']
    b=c23.c22_locked_predict(pack,f,masked_y,idx,args,parent,legacy)
    q,queue_stat=c23.queue_candidates(pack,f,idx,b,args,parent)
    winner23=q[win23] if win23 else b.copy()
    # Prespecified exploratory challengers, NEVER chosen by block labels.
    candidates={'C22_locked':np.asarray(b,float),
                'C23_selected':np.asarray(winner23,float),
                'C23_bad_lirf_flow_35_DESCRIPTIVE':q['c23_bad_lirf_flow_35'],
                'C23_bad_lirf_queue_35_DESCRIPTIVE':q['c23_bad_lirf_queue_35'],
                'C23_ltfm_queue_30_DESCRIPTIVE':q['c23_ltfm_queue_30']}
    assert all(len(v)==len(idx) and np.isfinite(v).all() and (v>=0).all()
               for v in candidates.values()),'Invalid predictions'
    yy=y[idx]
    months=f['_month'].iloc[idx].to_numpy()
    ap=f['airport'].iloc[idx].astype(str).to_numpy()
    good=parent._plausible_proxy(f,idx)
    records=[];details=[]
    for m in HOLDOUT:
        at=months==m
        mi=idx[at]
        d=f['_day'].iloc[mi].to_numpy()
        preds={k:np.asarray(v)[at] for k,v in candidates.items()}
        for j,(name,p) in enumerate(preds.items()):
            bt=parent.paired_bootstrap(y[mi],p,preds['C22_locked'],d,c.bootstrap,
                                       c.seed+101*j+int(m[-2:])*200)
            row={'outer':m,'candidate':name,
                 'selected_before_holdout':name=='C23_selected',
                 **metrics(y[mi],p),**bt}
            records.append(row)
        details.extend(diagnostics(f,y,mi,preds,parent,m))
        mainrow=metrics(y[mi],preds['C23_selected'])
        print(f'{m}: C22={metrics(y[mi],preds["C22_locked"])["rmse"]:.3f} '
              f'C23_selected={mainrow["rmse"]:.3f} '
              f'bad_LIRF={int(((ap=="LIRF")&(~good)&(months==m)).sum())} '
              f'LTFM_gt3600={int(((ap=="LTFM")&(yy>3600)&(months==m)).sum())}',flush=True)
    c.out_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(c.out_dir/'ranking_month_block_candidates_2025_DESCRIPTIVE.csv',index=False)
    pd.DataFrame(details).to_csv(c.out_dir/'ranking_month_block_diagnostics_2025_DESCRIPTIVE.csv',index=False)
    pd.DataFrame(selrows).to_csv(c.out_dir/'october_select_candidates.csv',index=False)
    if c.save_outer_predictions:
        out=pd.DataFrame({'outer':months,'utc_mvt':f['_timestamp'].iloc[idx].to_numpy(),
                          'airport':ap,'proxy_ok':good,'y_true':yy})
        if parent.ID in raw:out.insert(0,parent.ID,raw[parent.ID].iloc[idx].to_numpy())
        for name,v in candidates.items():out['pred_'+name]=v
        try:out.to_parquet(c.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:out.to_csv(c.out_dir/'blocked_predictions_2025_RESEARCH_ONLY.csv',index=False)
    pooled={k:metrics(yy,v) for k,v in candidates.items()}
    report={'campaign':26,'research_only':True,'no_2026':True,
      'design':'retrospective 2025 leave-four-month-block-out; NOT forward holdout',
      'blocked_months':list(HOLDOUT),'fit_months':list(FIT_MONTHS),
      'selector_month':SELECT_MONTH,'selector_fit_months':['2025-03','2025-04','2025-05','2025-08','2025-09'],
      'rows':{k:len(v) for k,v in cuts.items()},
      'c22_select':c22name,'c23_select':win23 or 'C22_locked',
      'main_predictions':pooled,
      'source_files':sources,'script_sha256':sha256_file(__file__),
      'parent_sha256':sha23,'parent22_sha256':sha22,'parent21_sha256':sha21,
      'preset':c.preset,'queue_state':queue_stat,
      'notes':['Model policy chosen on October 2025, before blocked months are scored.',
               'No Jan/Feb/Jun/Jul 2025 target in FIT, SELECT or lagged label priors.',
               'Non-chronological blocked validation; NOT a claim of fresh holdout.',
               'Only four months in ranking 2026; targets not available, not loaded.',
               'Do NOT deploy challenge scores chosen after looking at this report.']}
    (c.out_dir/'summary.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('POOLED 2025 BLOCK (not forward):',
          {k:round(v['rmse'],3) for k,v in pooled.items()},flush=True)
    print('Research output:',c.out_dir.resolve(),flush=True)
    print('NO 2026 READ / NO FINAL REFIT / NO SUBMISSION',flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssertionError,ValueError,RuntimeError,ImportError,FileNotFoundError,KeyError) as e:
        print(f'FATAL {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(2)
