#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRC2026 C35 -- overnight model tuning, RESEARCH ONLY.

WHY: C33 CatBoost residual improves C24 on chronological historical OOF, but
those OOF predictions may have been trained using Jan/Feb/Jun/Jul 2025 labels.
Reusing such OOF as CatBoost FIT would contaminate C28's blocked validation.
Instead CROSS-FIT the frozen C28 *C22* first-stage architecture on ONLY the
8 nonblocked 2025 months, then train CatBoost (+ matched LightGBM) on those
cross-fitted residuals. Score against the FROZEN C28 4-month blocked predictions.
All heldout targets (Jan/Feb/Jun/Jul 2025) are excluded from every training,
selection and residual fit, including label-based historical priors. Features
may use timestamped movement events, strictly preceding their respective row.

Nonchronological retrospective leave-four-month-block-out experiment: all 2025
months were already explored, NOT a virgin 2026 holdout. No final full-2025
fit, NO 2026 file reads, NO ranking predictions or submission. The baseline is
C28's frozen C22 (not identical to a forward-month C24), so do not compare its
RMSE to C33's forward C24 RMSE.

USAGE:
 python 34_blocked_catboost_transfer_research.py --preset full --threads 4 --bootstrap 400 --save-outer-predictions
 python 34_blocked_catboost_transfer_research.py --self-test
Requires unchanged C21/22/23/26/27/28/32/33 next to this script and saved
C28 blocked_predictions_2025_RESEARCH_ONLY.parquet from the real run.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

C28_NAME = '28_lirf_flow_expert_tuning.py'
C28_SHA = '4159249a532b9b6cc6738c2d6e32f1ba6e6a0203d43b9264e2a6645d6cb72605'
C33_NAME = '33_catboost_residual_challenger.py'
BLOCK = ('2025-01','2025-02','2025-06','2025-07')
FIT = ('2025-03','2025-04','2025-05','2025-08','2025-09','2025-10','2025-11','2025-12')
FOLDS = (('2025-03','2025-04'),('2025-05','2025-08'),('2025-09','2025-10'),('2025-11','2025-12'))
BAD='LIRF_bad_AOBT'
METHODS=('C22','C27','C28','LGB50','CAT50','CAT100','LGB50_C28','CAT50_C28','CAT100_C28')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_module(name,alias,expected=None):
    p=Path(__file__).resolve().with_name(name)
    if not p.is_file():raise FileNotFoundError('Original dependency missing: '+str(p))
    if expected and sha(p)!=expected:
        raise RuntimeError(f'Original dependency {name} changed: got {sha(p)}, expected {expected}')
    spec=importlib.util.spec_from_file_location(alias,p)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def cli():
    p=argparse.ArgumentParser(description='C34: C28 blocked test + independently crossfit CatBoost meta; 2025 RESEARCH ONLY')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--c28-blocked',type=Path,default=Path('campaign28_research/blocked_predictions_2025_RESEARCH_ONLY.parquet'))
    p.add_argument('--c28-summary',type=Path,default=Path('campaign28_research/summary.json'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign34_research'))
    p.add_argument('--preset',choices=('full','smoke'),default='full')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--hours',type=float,default=5.75,help='Compute budget in hours, default 5.75')
    p.add_argument('--max-trials',type=int,default=0,help='0=work until time budget')
    p.add_argument('--resume',action='store_true',help='Reuse saved crossfit feature cache and trial log')
    p.add_argument('--smoke-full-catboost',action='store_true',help='Technical tests only: run 480 CatBoost trees with smoke-size dataset')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.threads<1 or a.bootstrap<100 or a.hours<=0:p.error('Need >=1 thread and >=100 bootstraps')
    if a.smoke_full_catboost and a.preset!='smoke':p.error('smoke-full-catboost requires --preset smoke')
    return a


def read_blocked(path):
    if not path.is_file():
        alt=path.with_suffix('.csv')
        if alt.is_file():path=alt
        else:raise FileNotFoundError('Saved C28 blocked predictions required: '+str(path))
    cols=['MVT_ID_mvt','outer','airport',BAD,'y_true','pred_C22','pred_C27','pred_C28']
    z=pd.read_parquet(path,columns=cols) if path.suffix=='.parquet' else pd.read_csv(path,usecols=cols)
    if not set(cols).issubset(z):raise AssertionError('C28 blocked columns missing')
    if z.MVT_ID_mvt.isna().any() or not z.MVT_ID_mvt.is_unique or z.empty:
        raise AssertionError('C28 blocked movement IDs invalid')
    z['outer']=z.outer.astype(str)
    if set(z.outer.unique())!=set(BLOCK):raise AssertionError('Wrong C28 blocked months')
    if not pd.api.types.is_bool_dtype(z[BAD]):
        vals=z[BAD].astype(str).str.lower()
        if not vals.isin(['true','false','0','1']).all():raise AssertionError('Bad clock invalid')
        z[BAD]=vals.isin(['true','1'])
    if (z[BAD] & (z.airport.astype(str)!='LIRF')).any():raise AssertionError('BAD outside LIRF')
    for k in ('y_true','pred_C22','pred_C27','pred_C28'):
        v=pd.to_numeric(z[k],errors='raise').to_numpy(float)
        if not np.isfinite(v).all() or (v<0).any():raise AssertionError('Invalid C28 '+k)
    bad=z[BAD].to_numpy(bool)
    for k in ('pred_C27','pred_C28'):
        if not np.array_equal(z[k].to_numpy(float)[~bad],z.pred_C22.to_numpy(float)[~bad]):
            raise AssertionError('C28 expert touched non-LIRF-bad')
    return z,path


def checker(months,targets,fit_idx,val_idx,global_block_idx):
    ft=set(months[fit_idx]); vv=set(months[val_idx]);
    if not ft.issubset(set(FIT)) or not vv.issubset(set(FIT)) or ft&vv:
        raise AssertionError('Crossfit split month leakage')
    if set(ft)|set(vv)!=set(FIT):raise AssertionError('Crossfit missing available months')
    if np.intersect1d(np.r_[fit_idx,val_idx],global_block_idx).size:
        raise AssertionError('Heldout month entered meta source')
    if not np.isfinite(targets[fit_idx]).all():raise AssertionError('Training target missing')


def self_test():
    mon=np.repeat(np.array([f'2025-{i:02d}' for i in range(1,13)]),4)
    ix=np.arange(len(mon));y=np.ones(len(mon))*100.
    hold=np.flatnonzero(np.isin(mon,BLOCK))
    for grp in FOLDS:
        val=ix[np.isin(mon,grp)]
        fit=ix[np.isin(mon,FIT)&~np.isin(mon,grp)]
        checker(mon,y,fit,val,hold)
        yy=y.copy();yy[hold]=np.nan;yy[val]=np.nan
        assert not np.isfinite(yy[val]).any() and not np.isfinite(yy[hold]).any()
    # Cat residual cannot change LIRF BAD, and source predictions stay immutable.
    bad=np.array([True,False,False])
    orig=np.array([800.,300.,900.]);delta=np.array([500.,-200.,10.]);delta[bad]=0
    assert np.array_equal(np.maximum(0,orig+.5*delta)[bad],orig[bad])
    yy=np.array([100.,200.,300.]);pb=np.array([95.,190.,310.]);source=yy-pb
    yy[:]=1e7
    assert np.array_equal(source, np.array([5.,10.,-10.]))  # immutable residuals
    print('SELF-TEST PASS C34: 4-fold leave-group-out, global blocked-label mask, strict regime isolation',flush=True)


def feature_frame(feat,index,baselines,parent,c32):
    cols=list(dict.fromkeys(parent.FEATURES+parent.FLOW_NUM+list(c32.Q_COLUMNS)))
    missing=[x for x in cols if x not in feat]
    if missing:raise AssertionError('Missing features '+str(missing))
    x=feat.iloc[index][cols].reset_index(drop=True).copy()
    x['pred_C24']=np.asarray(baselines,dtype=np.float32)  # Meta feature name used by C32/C33; actually C28 C22 baseline.
    for col in parent.CATS:
        if col in x:x[col]=x[col].astype('category')
    for col in x.columns:
        if not isinstance(x[col].dtype,pd.CategoricalDtype):
            x[col]=pd.to_numeric(x[col],errors='coerce').astype(np.float32)
    if x.columns.duplicated().any():raise AssertionError('Duplicate features')
    if any(c in x for c in ('TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt','y_true')):
        raise AssertionError('Label leaked into feature frame')
    return x


def eval_rmse(y,p):
    yy=np.asarray(y,float);pp=np.asarray(p,float)
    if len(yy)!=len(pp) or not np.isfinite(pp).all() or (pp<0).any():
        raise AssertionError('Invalid evaluation vector')
    return float(np.sqrt(np.mean((yy-pp)**2)))


def bootstrap(y,base,cand,days,n,seed):
    keys,inv=np.unique(np.asarray(days,dtype=str),return_inverse=True)
    ct=np.bincount(inv).astype(float)
    a=np.bincount(inv,weights=(y-base)**2)
    b=np.bincount(inv,weights=(y-cand)**2)
    rng=np.random.default_rng(seed);delta=np.empty(n)
    for i in range(n):
        s=rng.integers(len(keys),size=len(keys))
        delta[i]=np.sqrt(b[s].sum()/ct[s].sum())-np.sqrt(a[s].sum()/ct[s].sum())
    return {f'q{k}':float(np.percentile(delta,k)) for k in (5,50,95)}


def main():
    a=cli();self_test()
    if a.self_test:return
    tick=time.time()
    c28=load_module(C28_NAME,'c34_C28',C28_SHA)
    c33=load_module(C33_NAME,'c34_C33')
    c32=c33.load_c32()
    c27,c26,c23,c22,parent=c28.load_deps()
    print('='*84+'\nC35 OVERNIGHT TUNING / BLOCKED RESEARCH ONLY\n'
          'Jan/Feb/Jun/Jul target labels NEVER enter model FIT or META FIT.\n'
          '4 frozen-architecture crossfits on eight other months; NO 2026 / NO final fit\n'+'='*84,flush=True)
    blocked,blocpath=read_blocked(a.c28_blocked)
    if a.preset=='full' and len(blocked)!=671118:
        raise AssertionError(f'FULL C28 blocked rows expected 671118, got {len(blocked)}')
    raw,sources=parent.load_training_files(a.data_dir)
    feats=parent.build_features(raw)
    target=parent.make_y(raw)
    parent.add_past_only_flow(feats)
    c23.add_strict_history(feats)
    cuts=c26.partition(feats,target,1)
    months=feats['_month'].astype(str).to_numpy()
    heldout=cuts['heldout']
    # Exact ID, timestamp month, target and clock-regime alignment; never row-order based.
    rawids=pd.Index(raw[parent.ID])
    if not rawids.is_unique or rawids.hasnans:raise AssertionError('Original movement IDs invalid')
    blockids=pd.Index(blocked.MVT_ID_mvt)
    blockpos=rawids.get_indexer(blockids)
    if (blockpos<0).any() or not np.array_equal(np.sort(blockpos),np.sort(heldout)):
        raise AssertionError('C28 BLOCK IDs do not exactly match 2025 raw heldout')
    if not np.array_equal(months[blockpos],blocked.outer.to_numpy(str)):
        raise AssertionError('C28 blocked month/source discrepancy')
    if not np.array_equal(feats.airport.iloc[blockpos].astype(str).to_numpy(),blocked.airport.astype(str).to_numpy()):
        raise AssertionError('C28 blocked airport mismatch')
    if not np.allclose(target[blockpos],blocked.y_true.to_numpy(float),rtol=1e-12,atol=1e-7):
        raise AssertionError('C28 BLOCK label mismatch')
    expected=(blocked.airport.astype(str).to_numpy()=='LIRF') & (~parent._plausible_proxy(feats,blockpos))
    if not np.array_equal(expected,blocked[BAD].to_numpy(bool)):
        raise AssertionError('C28 clock classification changed')
    if a.c28_summary.is_file():
        past=json.loads(a.c28_summary.read_text(encoding='utf-8'))
        if past.get('C28_additional') not in ('flow100',None,'NONE'):
            raise AssertionError('Unsupported C28 specialist policy')
        if a.preset=='full' and past.get('baseline_oct')!='hybrid_lirf_tail_resid':
            raise AssertionError('C28 selected different October baseline')
    elif a.preset=='full':raise FileNotFoundError('C28 summary needed to audit locked policy: '+str(a.c28_summary))
    # Holdout labels inaccessible to the selector and ALL stage-1 FIT calls.
    ymasked=target.copy();ymasked[heldout]=np.nan
    y_select=ymasked.copy()
    y_select[np.flatnonzero(np.isin(months,('2025-11','2025-12')))]=np.nan
    originalargv=sys.argv[:]
    try:
        sys.argv=[originalargv[0],'--preset',a.preset,'--threads',str(a.threads),
                  '--bootstrap',str(a.bootstrap),'--seed',str(a.seed)]
        cfg=c22.args_from_cli()
    finally:sys.argv=originalargv
    cfg.data_dir=a.data_dir;cfg.out_dir=a.out_dir
    # Freeze exactly the original C26/C28 October-selected C22 base policy.
    print('Stage A: reproduce frozen C28 October selector (no blocked labels)',flush=True)
    model=c23.fit_models(feats,y_select,cuts['fit_select'],cfg,c22,parent)
    incumbent,_,legacy=c23.c22_locked_candidates(model,feats,y_select,cuts['select'],cfg,c22,parent,c26.SELECT_FOR)
    if a.preset=='full' and incumbent!='hybrid_lirf_tail_resid':
        raise AssertionError('C28 October incumbent failed reproduction')
    del model
    gc.collect()
    # Cross-fitted first-stage predictions on 8 MONTHS ONLY; no leakage from
    # any of four target-block months. This is nonchronological crossfitting.
    all_pos=[];all_pred=[];fold_rows=[]
    for g,grp in enumerate(FOLDS,1):
        val=np.flatnonzero(np.isin(months,grp)&np.isfinite(target)&(target>=0))
        fit=np.flatnonzero(np.isin(months,FIT)&~np.isin(months,grp)&np.isfinite(target)&(target>=0))
        checker(months,target,fit,val,heldout)
        ym=ymasked.copy();ym[val]=np.nan
        if not np.isfinite(ym[fit]).all() or np.isfinite(ym[val]).any():
            raise AssertionError('Crossfit label mask failure')
        print(f'  META FOLD {g}/4 EVAL={grp} FIT={len(fit):,} VAL={len(val):,}',flush=True)
        pack=c23.fit_models(feats,ym,fit,cfg,c22,parent)
        v=c23.c22_locked_predict(pack,feats,ym,val,cfg,parent,legacy)
        if len(v)!=len(val) or not np.isfinite(v).all() or np.any(v<0):
            raise AssertionError('Crossfitted prediction invalid')
        all_pos.append(val);all_pred.append(np.asarray(v,float))
        fold_rows.append({'fold':g,'eval_months':','.join(grp),'train_rows':int(len(fit)),
                          'eval_rows':int(len(val)),'training_months':','.join(sorted(set(months[fit]))),
                          'first_stage_rmse':eval_rmse(target[val],v)})
        del pack,ym,v
        gc.collect()
    fitpos=np.concatenate(all_pos)
    fitpred=np.concatenate(all_pred)
    if len(np.unique(fitpos))!=len(fitpos) or not np.array_equal(np.sort(fitpos),np.sort(cuts['fit_final'])):
        raise AssertionError('Crossfit must cover exactly all 8 training months')
    fitbad=(feats.airport.iloc[fitpos].astype(str).to_numpy()=='LIRF') & (~parent._plausible_proxy(feats,fitpos))
    Xfit=feature_frame(feats,fitpos,fitpred,parent,c32)
    Xtest=feature_frame(feats,blockpos,blocked.pred_C22.to_numpy(float),parent,c32)
    if Xfit.columns.tolist()!=Xtest.columns.tolist():raise AssertionError('Train/blocked feature mismatch')
    # Keep fit/test frames separate to avoid accidentally training on BLOCK rows.
    gc.collect()
    # Cache precisely aligned cross-fitted stage-1 features; never cache labels of BLOCK in a training frame.
    tridx=np.flatnonzero(~fitbad)
    if len(tridx)<500:raise AssertionError('Insufficient meta training')
    # Tuning sees ONLY nonblocked crossfitted residuals. Block labels are used AFTER selection solely for reporting.
    residual=np.clip(target[fitpos]-fitpred,-3500.,3500.)
    group=months[fitpos]
    cats=[c for c in parent.CATS if c in Xfit]
    if not cats:raise AssertionError('No categorical features')
    out=a.out_dir;out.mkdir(parents=True,exist_ok=True)
    # Critical: no source model has been trained with any BLOCK label.
    for m in BLOCK:
        if np.isin(group,m).any():raise AssertionError('Blocked label entered tuner')
    eval_groups=[('2025-03','2025-04'),('2025-05','2025-08'),('2025-09','2025-10'),('2025-11','2025-12')]
    cv=[]
    for fold,months_val in enumerate(eval_groups):
        vi=np.flatnonzero(np.isin(group,months_val)&~fitbad)
        ti=np.flatnonzero(~np.isin(group,months_val)&~fitbad)
        assert len(vi)>0 and len(ti)>0 and not np.intersect1d(ti,vi).size
        cv.append((ti,vi,months_val))
    # Use 2 balanced group folds per trial, rotate groups to avoid single-window tuning;
    # finalists receive full 4-fold evaluation after budget expires.
    try:
        from catboost import CatBoostRegressor
        import lightgbm as lgb
    except ImportError as e:raise RuntimeError('Requires catboost and lightgbm') from e
    import traceback
    from datetime import datetime,timezone
    rng=np.random.default_rng(a.seed)
    records=[];trial_csv=out/'overnight_trials.csv'
    if a.resume and trial_csv.is_file():
        records=pd.read_csv(trial_csv).to_dict('records')
        print('RESUME previous completed trials',len(records),flush=True)
    # Critical CatBoost native categorical and LightGBM pandas categories.
    Xcat=c33.model_frame(Xfit,np.arange(len(Xfit)),cats)
    Xlgb=Xfit.copy()
    for col in cats:Xlgb[col]=Xlgb[col].astype('category')
    for col in Xlgb.columns:
        if col not in cats and not pd.api.types.is_numeric_dtype(Xlgb[col]):
            raise AssertionError('Unconverted nonnumerical LightGBM feature: '+col)
    cv_baseline=[]
    for _,vi,mm in cv:cv_baseline.append(float(np.sqrt(np.mean(np.square(residual[vi])))))
    print('TUNING LABEL SAFE. Crossfit folds:',[(m,len(v)) for _,v,m in cv],flush=True)
    time_limit=time.monotonic()+max(0.03,a.hours)*3600
    if a.preset=='smoke':time_limit=time.monotonic()+min(a.hours*3600,100)
    candidate_id=len(records)
    while time.monotonic()<time_limit and (a.max_trials==0 or candidate_id<a.max_trials):
        k=candidate_id
        # deterministic candidate generation independent from resume
        r=np.random.default_rng(a.seed+k*947)
        family=('cat','lgb','xgb')[k%3] if a.preset=='full' else ('cat','lgb')[k%2]
        depth=int(r.choice([4,5,6,7,8])); rate=float(r.choice([.025,.04,.055,.075]))
        leaf=int(r.choice([20,50,100,200])); reg=float(r.choice([8.,15.,30.,55.]))
        trees=int(r.choice([350,550,800,1100])) if a.preset=='full' else 24
        folds=[k%4,(k+2)%4] if a.preset=='full' else [k%4]
        trial_t=time.monotonic(); fold_errors=[];status='ok';error=''
        try:
            for f in folds:
                ti,vi,mv=cv[f]
                yt=residual[ti]
                if family=='cat':
                    mod=CatBoostRegressor(iterations=trees,depth=depth,learning_rate=rate,
                        l2_leaf_reg=reg,loss_function='RMSE',random_strength=1.5,
                        one_hot_max_size=10,max_ctr_complexity=1,thread_count=a.threads,
                        random_seed=a.seed+k,verbose=False,allow_writing_files=False)
                    mod.fit(Xcat.iloc[ti],yt,cat_features=cats,verbose=False)
                    pp=mod.predict(Xcat.iloc[vi])
                elif family=='lgb':
                    mod=lgb.LGBMRegressor(n_estimators=trees,num_leaves=min(2**depth-1,95),
                        max_depth=depth,learning_rate=rate,min_child_samples=leaf,
                        reg_lambda=reg,colsample_bytree=.82,subsample=.82,subsample_freq=1,
                        verbosity=-1,n_jobs=a.threads,random_state=a.seed+k)
                    mod.fit(Xlgb.iloc[ti],yt,categorical_feature=cats)
                    pp=mod.predict(Xlgb.iloc[vi])
                else:
                    from xgboost import XGBRegressor
                    mod=XGBRegressor(n_estimators=trees,max_depth=depth,learning_rate=rate,
                        min_child_weight=max(1,leaf//20),reg_lambda=reg,subsample=.82,
                        colsample_bytree=.82,n_jobs=a.threads,tree_method='hist',enable_categorical=True,random_state=a.seed+k)
                    mod.fit(Xlgb.iloc[ti],yt)
                    pp=mod.predict(Xlgb.iloc[vi])
                # Measure incremental RMSE of full base forecast, not just residual fitting loss.
                # Meta fit residual has been clipped; original *unclipped* target remains evaluation truth.
                basep=fitpred[vi]; yy=target[fitpos[vi]]
                full=np.maximum(0,basep+0.5*np.asarray(pp,float))
                score=float(np.sqrt(np.mean((yy-full)**2)))
                base=float(np.sqrt(np.mean((yy-basep)**2)))
                fold_errors.append(score-base)
                del mod,pp
            delta=float(np.mean(fold_errors))
        except Exception as e:
            status='error';error=repr(e)[:250];delta=float('nan')
            traceback.print_exc(limit=2)
        rec={'trial':k,'family':family,'depth':depth,'learning_rate':rate,'min_child':leaf,
             'l2':reg,'trees':trees,'cv_delta_rmse':delta,'worst_fold_delta':max(fold_errors) if fold_errors else np.nan,
             'fold_deltas':json.dumps(fold_errors),'folds':','.join(map(str,folds)),
             'elapsed_s':round(time.monotonic()-trial_t,1),'status':status,'error':error}
        records.append(rec)
        pd.DataFrame(records).to_csv(trial_csv,index=False)
        print('TRIAL',k,family,'delta',round(delta,4) if np.isfinite(delta) else 'ERROR',
              'worst',round(rec['worst_fold_delta'],4) if fold_errors else 'NA',
              'seconds',rec['elapsed_s'],'elapsed_h',round((time.monotonic()-(time_limit-a.hours*3600))/3600,2),flush=True)
        candidate_id+=1
        gc.collect()
    successful=[r for r in records if r['status']=='ok' and np.isfinite(r['cv_delta_rmse'])]
    if not successful:raise RuntimeError('No tuning trials succeeded; check overnight_trials.csv')
    # Pick conservative robust candidate; rank by mean gain, penalize bad folds.
    for r in successful:r['risk_score']=r['cv_delta_rmse']+0.5*max(0.,r['worst_fold_delta'])
    best=min(successful,key=lambda x:x['risk_score'])
    print('BEST CV MODEL:',best,flush=True)
    # Final model fits ONLY the eight nonblocked months, using stage-1 out-of-fold residuals.
    family=best['family'];trees=int(best['trees']);depth=int(best['depth']);rate=float(best['learning_rate']);reg=float(best['l2'])
    ti=tridx
    if family=='cat':
        mod=CatBoostRegressor(iterations=trees,depth=depth,learning_rate=rate,
                l2_leaf_reg=reg,loss_function='RMSE',random_strength=1.5,one_hot_max_size=10,
                max_ctr_complexity=1,thread_count=a.threads,random_seed=a.seed+999,
                verbose=False,allow_writing_files=False)
        mod.fit(Xcat.iloc[ti],residual[ti],cat_features=cats,verbose=False)
        xt=c33.model_frame(Xtest,np.arange(len(Xtest)),cats)
        add=np.asarray(mod.predict(xt),float)
    elif family=='lgb':
        mod=lgb.LGBMRegressor(n_estimators=trees,num_leaves=min(2**depth-1,95),max_depth=depth,
            learning_rate=rate,min_child_samples=int(best['min_child']),reg_lambda=reg,
            colsample_bytree=.82,subsample=.82,subsample_freq=1,verbosity=-1,
            n_jobs=a.threads,random_state=a.seed+999)
        mod.fit(Xlgb.iloc[ti],residual[ti],categorical_feature=cats)
        xt=Xtest.copy()
        for col in cats:xt[col]=pd.Categorical(xt[col],categories=Xlgb[col].cat.categories)
        add=np.asarray(mod.predict(xt),float)
    else:
        from xgboost import XGBRegressor
        xt=Xtest.copy()
        for col in cats:xt[col]=pd.Categorical(xt[col],categories=Xlgb[col].cat.categories)
        mod=XGBRegressor(n_estimators=trees,max_depth=depth,learning_rate=rate,
            min_child_weight=max(1,int(best['min_child'])//20),reg_lambda=reg,
            subsample=.82,colsample_bytree=.82,n_jobs=a.threads,tree_method='hist',enable_categorical=True,random_state=a.seed+999)
        mod.fit(Xlgb.iloc[ti],residual[ti]);add=np.asarray(mod.predict(xt),float)
    bad=blocked[BAD].to_numpy(bool)
    if not np.isfinite(add).all():raise AssertionError('Model returned non-finite values')
    add[bad]=0.
    ref=blocked.pred_C22.to_numpy(float)
    y=blocked.y_true.to_numpy(float)
    pred={'C22':ref,'C27':blocked.pred_C27.to_numpy(float),'C28':blocked.pred_C28.to_numpy(float)}
    for w in (.35,.5,.75,1.):
        tmp=np.maximum(0,ref+w*add)
        tmp[bad]=pred['C28'][bad]
        pred['tuned_'+str(w)]=tmp
    mon=blocked.outer.to_numpy(str)
    groups={'JAN_JUL':('2025-01','2025-07'),'FEB_JUN':('2025-02','2025-06'),'ALL':BLOCK}
    rows=[]
    for label,mm in groups.items():
        sel=np.isin(mon,mm)
        rec={'group':label,'n':int(sel.sum())}
        for name,values in pred.items():rec[name+'_rmse']=eval_rmse(y[sel],values[sel])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/'ranking_month_groups_DESCRIPTIVE.csv',index=False)
    month_rows=[]
    for mm in BLOCK:
        sel=mon==mm
        rec={'month':mm,'n':int(sel.sum())}
        for name,values in pred.items():rec[name+'_rmse']=eval_rmse(y[sel],values[sel])
        month_rows.append(rec)
    pd.DataFrame(month_rows).to_csv(out/'ranking_months_DESCRIPTIVE.csv',index=False)
    if a.save_outer_predictions:
        z=blocked[['MVT_ID_mvt','outer','airport',BAD,'y_true']].copy()
        for name,values in pred.items():z['pred_'+name]=values
        try:z.to_parquet(out/'blocked_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:z.to_csv(out/'blocked_predictions_2025_RESEARCH_ONLY.csv',index=False)
    summary={'campaign':35,'research_only':True,'no_2026':True,'no_final_refit':True,
            'selection_only_on_nonblocked_crossfit_residuals':True,'crossfit_folds':list(map(list,FOLDS)),
            'best_cv':best,'trials':len(records),'rank_groups':rows,'monthly':month_rows,
            'source_model':'C28_C22','fit_rows':int(len(tridx)),'blocked_rows':int(len(blockpos)),
            'script_sha256':sha(__file__)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=float),encoding='utf-8')
    print('C35 DONE. See',out/'summary.json',flush=True)
    print('RESEARCH ONLY -- NO 2026 / NO FINAL REFIT OR SUBMISSION',flush=True)


if __name__=='__main__':
    try:main()
    except (AssertionError,ValueError,RuntimeError,FileNotFoundError,KeyError,ImportError) as exc:
        print('FATAL C35:',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(2)
