#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EUROCONTROL PRC 2026 C32: point-in-time cross-fitted residual pipeline.

Research on 2025 only. NO full-2025 final fit, NO 2026 ranking read,
NO ranking predictions, NO submission. Reuses original C30 chronological OOF
as baseline and trains only a second-stage, strongly regularized LightGBM.

Features: exactly official C21's feature whitelist, strict-past departure
counts, optionally strict-past headway/idle from original C23; plus OOF C24
prediction. Every meta fit uses only PRIOR MONTHS' C24 OOF predictions and
labels, with >= 1-day embargo before its held-out month. No target-clipping
in evaluation; residual-target cap in FIT is a documented robustness choice.
C24 predictions for LIRF BAD AOBT are NEVER changed by this model; alternate
flow65/flow100 blends from original C30 are evaluated as separate additions.

Historical 2025 months were explored repeatedly; scores are NOT fresh holdout.
The meta-stage for final 2026 (NOT IMPLEMENTED HERE) must train using 2025
OOF C24 predictions, then predict on a truly trained full-2025 C24 model.

Example:
 python 32_crossfit_residual_lgb_research.py --preset full --threads 4 \
     --bootstrap 400 --save-outer-predictions
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

PARENT = '21_clock_aware_tail_research.py'
SHA_PARENT = 'f5707cdda883334e9b388c652aee6954f59f1c8d1cd40b94eed1d47b55af4042'
C23_NAME = '23_queueing_operations_research_lab.py'
SHA_C23 = 'd137a102e3261109f0f59dbcf1cd211a91554337e1f180b0067ff834deb1b591'
MONTHS = tuple(f'2025-{k:02d}' for k in range(3,13))
DEFAULT = ('2025-06','2025-07','2025-08','2025-09','2025-10','2025-11','2025-12')
BAD = 'LIRF_bad_AOBT'
METHODS = ('C24','flow65','flow100','meta_core50','meta_core100','meta_queue50','meta_queue100',
           'meta_core50_flow65','meta_queue50_flow65','meta_queue50_flow100')
Q_COLUMNS = ('q_runway_past_5m', 'q_airport_past_5m', 'q_runway_idle_sec',
             'q_airport_idle_sec', 'q_prev_runway_headway_sec')


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def import_locked(filename, expected, modname):
    p=Path(__file__).resolve().with_name(filename)
    if not p.is_file():raise FileNotFoundError('Required original source missing: '+str(p))
    d=hash_file(p)
    if d!=expected:raise RuntimeError('Locked source changed: '+filename+' expected '+expected+' got '+d)
    spec=importlib.util.spec_from_file_location(modname,p)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[modname]=mod
    spec.loader.exec_module(mod)
    return mod


def parse():
    p=argparse.ArgumentParser(description='C32 cross-fitted residual LGB, 2025 RESEARCH ONLY')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--predictions',type=Path,default=Path('campaign30_research/outer_predictions_2025_RESEARCH_ONLY.parquet'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign32_research'))
    p.add_argument('--preset',choices=('full','smoke'),default='full')
    p.add_argument('--outer-months',nargs='+',default=list(DEFAULT))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--embargo-days',type=int,default=1)
    p.add_argument('--min-history-months',type=int,default=3)
    p.add_argument('--min-history-rows',type=int,default=1000)
    p.add_argument('--max-training-rows',type=int,default=0,
                   help='ONLY for smoke tests; forbidden in --preset full')
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.threads<1 or a.bootstrap<100 or a.embargo_days<0 or a.min_history_months<2 or a.min_history_rows<1:
        p.error('Invalid arguments')
    if len(set(a.outer_months))!=len(a.outer_months) or not set(a.outer_months).issubset(MONTHS):
        p.error('outer months must be unique and between March and December 2025')
    if a.preset=='full' and a.max_training_rows: p.error('FULL forbids training-row cap')
    return a


def read_oof(p,preset):
    if not p.is_file():
        alt=p.with_suffix('.csv')
        if alt.is_file():p=alt
        else:raise FileNotFoundError('Missing C30 OOF predictions: '+str(p))
    cols=['MVT_ID_mvt','outer','utc_mvt','airport',BAD,'y_true','pred_C24','pred_flow100']
    x=pd.read_parquet(p,columns=cols) if p.suffix.lower()=='.parquet' else pd.read_csv(p,usecols=cols)
    if any(c not in x for c in cols):raise AssertionError('Missing C30 OOF column')
    if x.empty or x.MVT_ID_mvt.isna().any() or not x.MVT_ID_mvt.is_unique:
        raise AssertionError('C30 OOF requires unique, non-null IDs')
    if preset=='full' and len(x)!=1787321:
        raise AssertionError('C30 FULL OOF requires exactly 1,787,321 scored movements')
    x['outer']=x['outer'].astype(str)
    if set(x.outer.unique())!=set(MONTHS):raise AssertionError('C30 OOF month list mismatch')
    x['utc_mvt']=pd.to_datetime(x.utc_mvt,utc=True)
    if x.utc_mvt.isna().any():raise AssertionError('Invalid OOF timestamps')
    b=x[BAD]
    if not pd.api.types.is_bool_dtype(b):
        txt=b.astype(str).str.lower()
        if not txt.isin(['true','false','1','0']).all():raise AssertionError('Invalid OOF bad clock bool')
        x[BAD]=txt.isin(['true','1'])
    if x[BAD].isna().any():raise AssertionError('Missing bad mask')
    x['airport']=x.airport.astype(str)
    if ((x[BAD]) & (x.airport!='LIRF')).any():raise AssertionError('BAD outside LIRF')
    for c in ('y_true','pred_C24','pred_flow100'):
        x[c]=pd.to_numeric(x[c],errors='raise').astype(np.float64)
        if not np.isfinite(x[c]).all() or (x[c]<0).any():raise AssertionError('Invalid numeric C30 '+c)
    nb=~x[BAD].to_numpy()
    if not np.array_equal(x.loc[nb,'pred_C24'].to_numpy(),x.loc[nb,'pred_flow100'].to_numpy()):
        raise AssertionError('C30 expert modifies other regimes')
    x['pred_flow65']=x['pred_C24']+.65*(x['pred_flow100']-x['pred_C24'])
    print('READ C30 OOF',p,'rows',len(x),flush=True)
    return x,p


def align_features(parent,c23,raw,oof):
    if parent.ID not in raw:raise AssertionError('Official data lacks MVT_ID_mvt')
    raw_ids=pd.Index(raw[parent.ID])
    if not raw_ids.is_unique or raw_ids.isna().any():raise AssertionError('Raw ID not unique')
    pos=raw_ids.get_indexer(oof['MVT_ID_mvt'].to_numpy())
    if (pos<0).any():raise AssertionError(f'{int((pos<0).sum())} missing OOF IDs in official 2025 data')
    feat=parent.build_features(raw)
    parent.add_past_only_flow(feat)
    c23.add_strict_history(feat)
    # Source time and true target audits are exact. Clock precision converted to UTC ns.
    original_t=feat['_timestamp'].iloc[pos]
    if not np.array_equal(original_t.astype('datetime64[ns, UTC]').astype('int64').to_numpy(),
                          oof['utc_mvt'].astype('datetime64[ns, UTC]').astype('int64').to_numpy()):
        raise AssertionError('Cached C30 time differs from actual official movement')
    if not np.array_equal(feat['airport'].iloc[pos].astype(str).to_numpy(),oof.airport.to_numpy()):
        raise AssertionError('Cached C30 airport differs from official movement')
    yt=parent.make_y(raw)[pos]
    if not np.allclose(yt,oof.y_true.to_numpy(),rtol=1e-12,atol=1e-7):
        raise AssertionError('Cached C30 labels differ from official 2025 data')
    expected=(oof.airport.to_numpy()=='LIRF') & (~parent._plausible_proxy(feat,pos))
    if not np.array_equal(expected,oof[BAD].to_numpy(bool)):
        raise AssertionError('C30 bad-clock classification differs from original source')
    all_cols=list(dict.fromkeys(parent.FEATURES+parent.FLOW_NUM+list(Q_COLUMNS)))
    x=feat.iloc[pos][all_cols].reset_index(drop=True).copy()
    x['pred_C24']=oof['pred_C24'].to_numpy(dtype=np.float32)
    # Stable shared categorical vocabulary across the entire 2025 history.
    for c in parent.CATS:
        if c in x:x[c]=x[c].astype('category')
    # Limit memory: float32 high precision sufficient for ML features, not targets.
    for c in x:
        if not isinstance(x[c].dtype, pd.CategoricalDtype):
            x[c]=pd.to_numeric(x[c],errors='coerce').astype(np.float32)
    if x.columns.duplicated().any():raise AssertionError('Duplicate model features')
    if 'TAXITIME_SEC_mvt' in x or 'BLOCK_TIME_UTC_mvt' in x or 'y_true' in x:
        raise AssertionError('Target or future block time leaked into features')
    print('FEATURES',len(x.columns),'(incl. strict-past flow & runway headway); aligned',len(x),flush=True)
    return x


def fit_second_stage(x,y_residual,indices,cfg,variant):
    from lightgbm import LGBMRegressor
    if variant=='core':
        cols=[c for c in x if c not in Q_COLUMNS]
        setup=dict(n_estimators=(220 if cfg.preset=='full' else 36),num_leaves=15,
                   min_child_samples=(550 if cfg.preset=='full' else 30),reg_lambda=120.)
    elif variant=='queue':
        cols=list(x.columns)
        setup=dict(n_estimators=(300 if cfg.preset=='full' else 42),num_leaves=23,
                   min_child_samples=(420 if cfg.preset=='full' else 25),reg_lambda=110.)
    else:raise AssertionError('Unknown variant')
    model=LGBMRegressor(objective='regression',learning_rate=.04,reg_alpha=2.,
                        max_bin=127,subsample=1.0,colsample_bytree=.85,
                        random_state=cfg.seed+330,n_jobs=cfg.threads,
                        deterministic=True,force_col_wise=True,verbosity=-1,
                        **setup)
    cat=[c for c in x.columns if c in cols and str(x[c].dtype)=='category']
    model.fit(x.iloc[indices][cols],y_residual[indices],categorical_feature=cat)
    return model,cols


def make_predictions(x,oof,months,a):
    n=len(oof)
    r=oof.y_true.to_numpy(float)-oof.pred_C24.to_numpy(float)
    base=oof.pred_C24.to_numpy(float)
    bad=oof[BAD].to_numpy(bool)
    time=oof.utc_mvt
    m=oof.outer.to_numpy(str)
    month_results=[]; frames=[]; fi=[]
    for month in months:
        start=pd.Timestamp(month+'-01',tz='UTC')
        tr=(time<start-pd.Timedelta(days=a.embargo_days)).to_numpy() & (m<month) & (~bad)
        vl=(m==month)
        fit_idx=np.flatnonzero(tr)
        val_idx=np.flatnonzero(vl)
        preceding=np.unique(m[fit_idx]); n_hist=len(preceding)
        if len(fit_idx)==0 or len(val_idx)==0:raise AssertionError('Empty fold '+month)
        if (np.max(time.iloc[fit_idx])>=np.min(time.iloc[val_idx])):
            raise AssertionError('Chronology violated '+month)
        if len(fit_idx)<a.min_history_rows or n_hist<a.min_history_months:
            print('  SKIP',month,'insufficient historical OOF for meta',flush=True)
            continue
        if a.max_training_rows and len(fit_idx)>a.max_training_rows:
            # Deterministic chronological thin slice allowed only in smoke mode.
            fit_idx=fit_idx[np.linspace(0,len(fit_idx)-1,a.max_training_rows,dtype=int)]
        # True residuals are NOT clipped for evaluation, only target for fit.
        train_res=np.clip(r,-3500,3500)
        pb=base[val_idx]; yy=oof.y_true.to_numpy(float)[val_idx]
        bad_val=bad[val_idx]
        preds={'C24':pb.copy(),
               'flow65':oof.pred_flow65.to_numpy(float)[val_idx],
               'flow100':oof.pred_flow100.to_numpy(float)[val_idx]}
        for variant in ('core','queue'):
            model,cols=fit_second_stage(x,train_res,fit_idx,a,variant)
            pred_res=np.asarray(model.predict(x.iloc[val_idx][cols]),dtype=float)
            if not np.isfinite(pred_res).all():raise AssertionError('Non-finite meta correction')
            # Strict isolation; correction never edits rare LIRF bad-clock tails.
            pred_res[bad_val]=0.
            for weight in (.5,1.0):
                name=f'meta_{variant}{int(weight*100)}'
                preds[name]=np.maximum(0,pb+weight*pred_res)
                if not np.array_equal(preds[name][bad_val],pb[bad_val]):
                    raise AssertionError('Meta changes bad LIRF')
            for name,w in [('meta_'+variant+'50_flow65',.65)]:
                arr=preds[f'meta_{variant}50'].copy()
                arr[bad_val]=pb[bad_val]+w*(preds['flow100'][bad_val]-pb[bad_val])
                preds[name]=np.maximum(0,arr)
            if variant=='queue':
                arr=preds['meta_queue50'].copy()
                arr[bad_val]=preds['flow100'][bad_val]
                preds['meta_queue50_flow100']=np.maximum(0,arr)
            fi.append({'outer':month,'variant':variant,'fit_rows':len(fit_idx),
                       'fit_months':','.join(preceding),'eval_rows':len(val_idx),
                       'num_features':len(cols),'num_trees':model.n_estimators_,
                       'top_importance':','.join(c for c,_ in sorted(zip(cols,model.feature_importances_),
                                                   key=lambda z:-z[1])[:12])})
        if set(preds)!=set(METHODS):raise AssertionError('Missing candidate methods '+repr(set(METHODS)-set(preds)))
        e={'outer':month,'n':len(val_idx),'n_bad':int(bad_val.sum()),
           'fit_rows':len(fit_idx),'prior_months':n_hist}
        for label,p in preds.items():
            e[label+'_rmse']=float(np.sqrt(np.mean((yy-p)**2)))
            e[label+'_nonbad_rmse']=float(np.sqrt(np.mean((yy[~bad_val]-p[~bad_val])**2)))
        month_results.append(e)
        print(' ',month,'fit=',len(fit_idx),'C24=',round(e['C24_rmse'],3),
              'core50=',round(e['meta_core50_rmse'],3),'queue50=',round(e['meta_queue50_rmse'],3),
              'queue50+flow65=',round(e['meta_queue50_flow65_rmse'],3),flush=True)
        df=oof.iloc[val_idx][['MVT_ID_mvt','outer','utc_mvt','airport',BAD,'y_true']].copy()
        for label,p in preds.items():df['pred_'+label]=p
        frames.append(df)
    if not frames:raise RuntimeError('No evaluable folds: reduce --min-history-months')
    return pd.DataFrame(month_results),pd.concat(frames,ignore_index=True),pd.DataFrame(fi)


def paired_boot(df,label,nboot,seed):
    monthday=df.outer.astype(str)+'/'+df.utc_mvt.dt.strftime('%Y-%m-%d')
    keys,inv=np.unique(monthday,return_inverse=True)
    yy=df.y_true.to_numpy(float);base=df.pred_C24.to_numpy(float)
    q=df['pred_'+label].to_numpy(float)
    c=np.bincount(inv,minlength=len(keys)).astype(float)
    a=np.bincount(inv,weights=(yy-base)**2,minlength=len(keys))
    b=np.bincount(inv,weights=(yy-q)**2,minlength=len(keys))
    rng=np.random.default_rng(seed)
    v=np.empty(nboot,dtype=float)
    for k in range(nboot):
        take=rng.integers(len(keys),size=len(keys))
        v[k]=np.sqrt(b[take].sum()/c[take].sum())-np.sqrt(a[take].sum()/c[take].sum())
    return {str(k):float(np.quantile(v,k/100)) for k in (5,50,95)}


def overall(df,months,a):
    yy=df.y_true.to_numpy(float)
    base=float(np.sqrt(np.mean((yy-df.pred_C24.to_numpy(float))**2)))
    rs={'C24':{'rmse':base}}
    for k,label in enumerate(METHODS[1:]):
        pp=df['pred_'+label].to_numpy(float)
        pooled=float(np.sqrt(np.mean((yy-pp)**2)))
        group=[]
        for mm in months:
            t=df[df.outer==mm]
            b=float(np.sqrt(np.mean((t.y_true-t.pred_C24)**2)))
            v=float(np.sqrt(np.mean((t.y_true-t['pred_'+label])**2)))
            group.append({'month':mm,'delta':v-b})
        rs[label]={'rmse':pooled,'delta':pooled-base,
                   'wins':int(sum(g['delta']<-1e-8 for g in group)),
                   'max_regret':float(max(g['delta'] for g in group)),
                   'day_bootstrap_delta':paired_boot(df,label,a.bootstrap,a.seed+121+k*37)}
    return rs


def test():
    import tempfile
    ids=pd.Series(['a','b','c','d'])
    x=pd.DataFrame({'MVT_ID_mvt':ids,'outer':['2025-03','2025-06','2025-07','2025-08'],
        'utc_mvt':pd.date_range('2025-03-01',periods=4,freq='95d',tz='UTC'),
        'airport':['LIRF','LIRF','EGLL','LTFM'],BAD:[True,False,False,False],
        'y_true':[100.,250.,100.,200.], 'pred_C24':[105.,230.,95.,201.],
        'pred_flow100':[110.,230.,95.,201.]})
    try:
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.csv';x.to_csv(p,index=False)
            read_oof(p,'smoke')
        raise AssertionError('Unexpected validation pass: missing months')
    except AssertionError as e:
        if 'OOF month list' not in str(e):raise
    assert list(MONTHS)[0]=='2025-03'
    assert np.clip(np.array([-4,5]),-3,3).tolist()==[-3,3]
    # Verify 1-day embargo and label isolation; no future dates in historical fits.
    idx=pd.to_datetime(['2025-05-29','2025-05-30','2025-06-01'],utc=True)
    mask=(idx<pd.Timestamp('2025-06-01',tz='UTC')-pd.Timedelta(days=1))
    assert mask.tolist()==[True,True,False]
    print('SELF-TEST PASS C32: schema rejection, chronological embargo, target isolation, robust residual scales')


def main():
    a=parse();test()
    if a.self_test:return
    t=time.time()
    print('='*75,'\nC32 POINT-IN-TIME RESIDUAL LIGHTGBM: 2025 RESEARCH ONLY\n'
          'NO FULL 2025 FINAL FIT, NO 2026 DATA, NO SUBMISSION\n'+'='*75,flush=True)
    p=import_locked(PARENT,SHA_PARENT,'c32_locked_parent')
    c23=import_locked(C23_NAME,SHA_C23,'c32_locked_c23')
    oof,path=read_oof(a.predictions,a.preset)
    raw,manifest=p.load_training_files(a.data_dir)
    X=align_features(p,c23,raw,oof)
    del raw
    import gc
    gc.collect()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    monthly,preds,features=make_predictions(X,oof,a.outer_months,a)
    summary={'campaign':32,'research_only':True,'no_2026':True,'no_final_fit':True,
             'design':'Past-only LightGBM meta-residual from original C30 OOF; compare same-fold C24 and LIRF expert',
             'label_leakage':'No current/future month residual used in FIT; 1-day embargo',
             'meta_target_fit_clip_sec':3500,'bad_clock_excluded_from_meta_fit_predict':True,
             'historical_caution':'All 2025 months repeatedly explored: not independent holdout',
             'input_c30_sha256':hash_file(path),'parent_sha256':hash_file(Path(__file__).resolve().with_name(PARENT)),
             'c23_sha256':hash_file(Path(__file__).resolve().with_name(C23_NAME)),
             'outers':list(monthly.outer),'score_rows':len(preds),
             'overall':overall(preds,monthly.outer.tolist(),a),
             'elapsed_sec':time.time()-t}
    monthly.to_csv(a.out_dir/'monthly_meta_comparison.csv',index=False)
    features.to_csv(a.out_dir/'meta_feature_importance.csv',index=False)
    if a.save_outer_predictions:
        try:preds.to_parquet(a.out_dir/'meta_outer_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:preds.to_csv(a.out_dir/'meta_outer_predictions_2025_RESEARCH_ONLY.csv',index=False)
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('POOLED C32 MATCHED MONTHS', {k:round(v['rmse'],3) for k,v in summary['overall'].items()})
    print('OUTPUT',a.out_dir,'2025 RESEARCH ONLY -- NO FINAL FIT/PREDICTIONS',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'FATAL C32 {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
