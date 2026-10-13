#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EUROCONTROL PRC 2026 Campaign 23: research-only queueing and service capacity.

Requires UNMODIFIED Campaign 22 and Campaign 21 modules in the same directory.
Hash locks prevent accidental mixing of experiment versions.

No 2026 data is read; no ranking data, no submission, no full-2025 refit.
All held-out months in 2025 were already explored (DESCRIPTIVE evaluations).

Two complementary queueing hypotheses are evaluated against locked C22:
1. Empirical runway/airport SERVICE CAPACITY estimated from strictly earlier
   FIT departures, coupled to M/D/1 queue-delay proxies, runway headways,
   idle times, and utilization. All movement counts are strictly historical.
2. A supervised expert with those OR features, compared with a matched
   FLOW-ONLY control trained on the identical labels and model settings.
   A LIRF bad-clock specialist is also compared with its flow-only control.

Crucial: this is NOT observation of the true physical takeoff queue. Flow
of completed departures is a service proxy, NOT latent demand. We do NOT use
future movements or hidden taxi-out labels to compute queue states.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PARENT_NAME = '22_bad_clock_regime_research.py'
PARENT_SHA = '754ef9f9a0dc3b05ca3e662710af1cb7bb0e61add1cdb7d1b928c9352bc9d088'
QUEUE_COLUMNS = [
    'q_runway_past_5m','q_airport_past_5m',
    'q_runway_idle_sec','q_airport_idle_sec','q_prev_runway_headway_sec',
    'q_runway_capacity_15m','q_airport_capacity_15m',
    'q_runway_utilization','q_airport_utilization',
    'q_runway_md1_wait_sec','q_airport_md1_wait_sec',
    'q_runway_pressure','q_airport_pressure',
]
CANDIDATES = (
    'c23_global_queue_15',
    'c23_global_queue_30',
    'c23_lirf_queue_30',
    'c23_ltfm_queue_30',
    'c23_congested_queue_40',
    'c23_bad_lirf_queue_35',
    'c23_bad_lirf_flow_35',
    'c23_global_flow_control_30',
)


def load_deps():
    p = Path(__file__).resolve().with_name(PARENT_NAME)
    if not p.is_file():
        raise FileNotFoundError(f'Missing Campaign 22 dependency beside this script: {p}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != PARENT_SHA:
        raise RuntimeError(f'Campaign 22 hash mismatch; expected {PARENT_SHA}, got {digest}')
    spec = importlib.util.spec_from_file_location('campaign22_locked_for23', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    parent, parent_sha = mod.load_parent()
    return mod, parent, digest, parent_sha


def _ns(feat):
    ts = feat['_timestamp']
    if not bool(ts.notna().all()):
        raise ValueError('Missing event timestamp')
    vals = (ts.astype('datetime64[ns, UTC]')
            .astype('int64').to_numpy(dtype=np.int64, copy=False))
    if not np.all(vals > 0):
        raise ValueError('Invalid event timestamp')
    return vals


def add_strict_history(feat):
    """Past-only features from MVT timestamps. Ties never observe each other."""
    t = _ns(feat)
    ap = feat['airport'].astype(str)
    rw = feat['runway'].astype(str)
    n = len(feat)
    for group, prefix in [
        (ap, 'q_airport'), (ap.str.cat(rw, sep='||'),'q_runway')
    ]:
        codes = pd.factorize(group, sort=False)[0]
        order = np.lexsort((t, codes))
        ranked_codes = codes[order]
        edges = np.r_[0, np.flatnonzero(np.diff(ranked_codes))+1, n]
        count_5 = np.zeros(n, dtype=np.float32)
        idle = np.full(n, 7200.0, dtype=np.float32)
        prev_headway = np.full(n, 7200.0, dtype=np.float32)
        for lo,hi in zip(edges[:-1],edges[1:]):
            idx = order[lo:hi]
            tt = t[idx]
            past_upper = np.searchsorted(tt, tt, side='left')
            past_lower = np.searchsorted(tt, tt-300_000_000_000, side='left')
            count_5[idx] = past_upper-past_lower
            previous = past_upper-1
            valid = previous>=0
            if np.any(valid):
                vpos = np.flatnonzero(valid)
                gap = (tt[vpos]-tt[previous[vpos]])/1e9
                idle[idx[vpos]] = np.clip(gap, 0, 7200).astype(np.float32)
                # The *previous* completed runway interdeparture interval,
                # not the query gap. Previous predecessor is strictly earlier.
                former = np.searchsorted(tt,tt[previous[vpos]],side='left')-1
                ok = former>=0
                if ok.any():
                    vpos2=vpos[ok]
                    last_gap=(tt[previous[vpos2]]-tt[former[ok]])/1e9
                    prev_headway[idx[vpos2]]=np.clip(last_gap,0,7200).astype(np.float32)
        feat[prefix+'_past_5m'] = count_5
        feat[prefix+'_idle_sec'] = idle
        if prefix == 'q_runway':
            feat['q_prev_runway_headway_sec'] = prev_headway
    if not np.isfinite(feat[['q_airport_past_5m','q_runway_past_5m']].to_numpy()).all():
        raise AssertionError('Invalid strict history')


def _group_capacity(feat, fit_idx, keys, col):
    """Fit-only unsupervised high-flow quantile, shrinkage to a global cap."""
    x=feat.iloc[fit_idx][list(keys)+[col]].copy()
    vals=np.maximum(0,np.asarray(x[col],float))
    global_cap=max(2.0,float(np.quantile(vals,.985))+1.0)
    x[col]=vals
    grouped=x.groupby(list(keys),sort=False,dropna=False,observed=True)[col]
    capacity=(grouped.quantile(.985)+1.0).clip(lower=2.0)
    counts=grouped.size()
    # Sparse runways borrow capacity from global distribution.
    pooled=(capacity*counts/(counts+150) + global_cap*150/(counts+150))
    return dict(pooled.items()), global_cap


def fit_or_state(feat, fit_idx):
    if len(fit_idx)==0:raise AssertionError('Empty FIT for queue capacity')
    run, run_cap=_group_capacity(feat,fit_idx,['airport','runway'],'runway_past_15m')
    air, air_cap=_group_capacity(feat,fit_idx,['airport'],'dep_past_15m')
    # Groupby key for one key is scalar in pandas; for runway is a tuple.
    return {'runway':run,'airport':air,
            'fallback_runway':run_cap,'fallback_airport':air_cap}


def queue_matrix(feat, idx, state):
    """Analytic M/D/1 proxy computed from past-only departure SERVICE rate.

    rho := rolling departures / fitted empirical service capacity. This is
    NOT an unbiased estimator of latent utilization or waiting times. All
    pseudo-waits are bounded and given to the learner as optional features.
    """
    f=feat.iloc[idx]
    air=f['airport'].astype(str).to_numpy()
    run=f['runway'].astype(str).to_numpy()
    cap_run=np.fromiter((state['runway'].get((a,r),state['fallback_runway'])
                         for a,r in zip(air,run)),dtype=np.float64,count=len(idx))
    cap_air=np.fromiter((state['airport'].get(a,state['fallback_airport'])
                         for a in air),dtype=np.float64,count=len(idx))
    cr=np.maximum(2,cap_run); ca=np.maximum(2,cap_air)
    nr=np.asarray(f['runway_past_15m'],float)
    na=np.asarray(f['dep_past_15m'],float)
    rho_r=np.clip(nr/cr,0,.985)
    rho_a=np.clip(na/ca,0,.985)
    # mu = empirically fitted hourly capacity, via a 15-minute window.
    mu_r=cr/900.
    mu_a=ca/900.
    md1_run=np.clip(rho_r**2/(2*mu_r*(1-rho_r)),0,3600)
    md1_air=np.clip(rho_a**2/(2*mu_a*(1-rho_a)),0,3600)
    q=pd.DataFrame(index=f.index)
    q['q_runway_past_5m']=f['q_runway_past_5m'].to_numpy(dtype=np.float32)
    q['q_airport_past_5m']=f['q_airport_past_5m'].to_numpy(dtype=np.float32)
    q['q_runway_idle_sec']=f['q_runway_idle_sec'].to_numpy(dtype=np.float32)
    q['q_airport_idle_sec']=f['q_airport_idle_sec'].to_numpy(dtype=np.float32)
    q['q_prev_runway_headway_sec']=f['q_prev_runway_headway_sec'].to_numpy(dtype=np.float32)
    q['q_runway_capacity_15m']=cr.astype('float32')
    q['q_airport_capacity_15m']=ca.astype('float32')
    q['q_runway_utilization']=rho_r.astype('float32')
    q['q_airport_utilization']=rho_a.astype('float32')
    q['q_runway_md1_wait_sec']=md1_run.astype('float32')
    q['q_airport_md1_wait_sec']=md1_air.astype('float32')
    q['q_runway_pressure']=np.maximum(0,nr-cr).astype('float32')
    q['q_airport_pressure']=np.maximum(0,na-ca).astype('float32')
    assert q.columns.tolist()==QUEUE_COLUMNS
    assert np.isfinite(q.to_numpy()).all()
    return q


def assemble(feat,idx,state,features,parent):
    base=parent.matrix(feat,idx,features)
    q=queue_matrix(feat,idx,state)
    for name in QUEUE_COLUMNS:
        base[name]=q[name].to_numpy(dtype=np.float32)
    return base


def fit_models(feat,y,fit_idx,args,c22,parent):
    p22=c22.fit_bad_experts(feat,y,fit_idx,args,parent)
    fit_idx=p22['parent']['parent']['old']['incumbent']['fit_indices']
    state=fit_or_state(feat,fit_idx)
    print('    C23 fitting queue-augmented L2 and matched flow-only control',flush=True)
    fcols=parent.FEATURES+parent.FLOW_NUM
    xflow=parent.matrix(feat,fit_idx,fcols)
    xq=assemble(feat,fit_idx,state,fcols,parent)
    models={
        'flow_global': parent.fit_experiment_regressor(xflow,y[fit_idx],args),
        'queue_global': parent.fit_experiment_regressor(xq,y[fit_idx],args),
    }
    aps=feat['airport'].iloc[fit_idx].astype(str).to_numpy()
    bad=(aps=='LIRF') & (~parent._plausible_proxy(feat,fit_idx))
    bad_idx=fit_idx[bad]
    days=feat['_day'].iloc[bad_idx].nunique()
    active=len(bad_idx)>=args.c22_min_bad_rows and days>=args.c22_min_bad_days
    models['bad_flow']=None
    models['bad_queue']=None
    if active:
        no_cols=parent.FEATURES_NO_AOBT+parent.FLOW_NUM
        badflow=parent.matrix(feat,bad_idx,no_cols)
        badqueue=assemble(feat,bad_idx,state,no_cols,parent)
        models['bad_flow']=parent.fit_experiment_regressor(badflow,y[bad_idx],args,tail=True)
        models['bad_queue']=parent.fit_experiment_regressor(badqueue,y[bad_idx],args,tail=True)
    print(f'    C23 FIT={len(fit_idx):,}, LIRF bad={len(bad_idx):,}, '
          f'bad expert active={active}, fitted cap R={state["fallback_runway"]:.1f}'
          f'/15m A={state["fallback_airport"]:.1f}/15m',flush=True)
    return {'parent':p22,'models':models,'state':state,
            'info':{'fit_rows':len(fit_idx),'bad_rows':len(bad_idx),
                    'bad_active':active,'runway_capacity_global':state['fallback_runway'],
                    'airport_capacity_global':state['fallback_airport']}}


def c22_locked_candidates(pack,feat,y,indices,args,c22,parent,outer):
    """Pre-select the Campaign22 incumbent on previous month SELECT only."""
    old=pack['parent']
    c20s=parent.predict_campaign20(old['parent']['parent'],feat,y,indices,args)
    c20_name,c19_name,c20rows=parent.select_campaign20(
        feat,y,indices,c20s,args,outer)
    c21s=parent.c21_candidate_predictions(old['parent'],feat,y,indices,args,
                                          c20_name,c20s)
    c21_name,c21rows=parent.c21_select(feat,y,indices,c20_name,c21s,c20s,args,outer)
    base=(c20s[c21_name] if c21_name in c20s else c21s[c21_name])
    bads,status=c22.bad_candidates(old,feat,indices,base,args,parent)
    c22_name,c22rows=c22.select_bad(feat,y,indices,base,bads,args,parent,outer)
    selected=c22_name or c21_name
    c22pred=bads[c22_name] if c22_name else base
    return selected,c22pred,{'c19':c19_name,'c20':c20_name,'c21':c21_name,
                             'c22':c22_name,'c20rows':c20rows,'c21rows':c21rows,
                             'c22rows':c22rows}


def c22_locked_predict(pack,feat,y,indices,args,parent,info):
    old=pack['parent']
    c20=parent.predict_campaign20(old['parent']['parent'],feat,y,indices,args)
    c21=parent.c21_candidate_predictions(old['parent'],feat,y,indices,args,
                                         info['c20'],c20)
    base=c20[info['c21']] if info['c21'] in c20 else c21[info['c21']]
    if info['c22']:
        # Refit only FIT_OUTER; same locked candidate chosen on SELECT.
        import importlib
        c22=sys.modules['campaign22_locked_for23']
        bads,_=c22.bad_candidates(old,feat,indices,base,args,parent)
        return bads[info['c22']]
    return base


def queue_candidates(pack,feat,idx,base,args,parent):
    base=np.asarray(base,dtype=np.float64)
    aps=feat['airport'].iloc[idx].astype(str).to_numpy()
    plausible=parent._plausible_proxy(feat,idx)
    q=queue_matrix(feat,idx,pack['state'])
    xflow=parent.matrix(feat,idx,parent.FEATURES+parent.FLOW_NUM)
    xq=assemble(feat,idx,pack['state'],parent.FEATURES+parent.FLOW_NUM,parent)
    pflow=np.maximum(0,np.asarray(pack['models']['flow_global'].predict(xflow),float))
    pq=np.maximum(0,np.asarray(pack['models']['queue_global'].predict(xq),float))
    lirf=aps=='LIRF'
    ltfm=aps=='LTFM'
    bad=lirf & ~plausible
    high=(q['q_runway_utilization'].to_numpy()>=.80) & (q['q_runway_past_5m'].to_numpy()>=2)
    def replace(mask,blend,target):
        out=base.copy()
        out[mask]=np.maximum(0,(1-blend)*base[mask]+blend*target[mask])
        return out
    pred={
        'c23_global_queue_15':replace(np.ones(len(idx),bool),.15,pq),
        'c23_global_queue_30':replace(np.ones(len(idx),bool),.30,pq),
        'c23_lirf_queue_30':replace(lirf,.30,pq),
        'c23_ltfm_queue_30':replace(ltfm,.30,pq),
        'c23_congested_queue_40':replace(high,.40,pq),
        'c23_global_flow_control_30':replace(np.ones(len(idx),bool),.30,pflow),
    }
    badflow=base.copy(); badqueue=base.copy()
    if pack['models']['bad_flow'] is not None and bad.any():
        no_cols=parent.FEATURES_NO_AOBT+parent.FLOW_NUM
        bidx=idx[bad]
        xf=parent.matrix(feat,bidx,no_cols)
        xqr=assemble(feat,bidx,pack['state'],no_cols,parent)
        vflow=np.maximum(0,np.asarray(pack['models']['bad_flow'].predict(xf),float))
        vqueue=np.maximum(0,np.asarray(pack['models']['bad_queue'].predict(xqr),float))
        badflow[bad]=np.maximum(0,.65*base[bad]+.35*vflow)
        badqueue[bad]=np.maximum(0,.65*base[bad]+.35*vqueue)
    pred['c23_bad_lirf_flow_35']=badflow
    pred['c23_bad_lirf_queue_35']=badqueue
    assert set(pred)==set(CANDIDATES)
    assert all(np.isfinite(v).all() and (v>=0).all() for v in pred.values())
    return pred,{'query_LIRF':int(lirf.sum()),'query_LIRF_bad':int(bad.sum()),
                 'query_high_runway_rho':int(high.sum()),
                 'mean_runway_rho':float(q['q_runway_utilization'].mean()),
                 'mean_queue_delay_proxy':float(q['q_runway_md1_wait_sec'].mean())}


def select_new(feat,y,idx,base,preds,args,parent,outer):
    yy=y[idx]
    days=feat['_day'].iloc[idx].to_numpy()
    left,right=parent.split_two_halves(feat,idx)
    mask=np.isin(idx,left)
    ref_a=parent.score(yy[mask],base[mask])['rmse']
    ref_b=parent.score(yy[~mask],base[~mask])['rmse']
    records=[]; qualified=[]
    for j,name in enumerate(CANDIDATES):
        p=preds[name]
        if np.array_equal(p,base):
            boot={'delta_rmse':0.,'q05_delta':0.,'q50_delta':0.,
                  'q90_delta':0.,'q95_delta':0.,'n_days':int(len(np.unique(days)))}
        else:
            boot=parent.paired_bootstrap(yy,p,base,days,args.bootstrap,
                                          args.seed+130001+j*127)
        a=parent.score(yy[mask],p[mask])['rmse']-ref_a
        b=parent.score(yy[~mask],p[~mask])['rmse']-ref_b
        good=(boot['delta_rmse']<=-1.0 and max(a,b)<=1.0
              and boot['q95_delta']<0.)
        record={'outer':outer,'select':parent.months_before(outer),
                'candidate':name,'qualified':bool(good),
                'half1_delta_sec':float(a),'half2_delta_sec':float(b),
                **parent.score(yy,p),**boot}
        records.append(record)
        if good: qualified.append(record)
    # Conservative candidate selection on SELECT only; no OUTER peeking.
    choice=(sorted(qualified,key=lambda r:(max(r['half1_delta_sec'],r['half2_delta_sec']),
                                           r['rmse'],r['candidate']))[0]['candidate']
            if qualified else None)
    return choice,records


def self_test(parent,c22):
    c22.self_test(parent)
    d=pd.DataFrame({
        parent.TAKEOFF:pd.to_datetime([
            '2025-07-01T09:55:00Z','2025-07-01T10:00:00Z',
            '2025-07-01T10:00:00Z','2025-07-01T10:04:00Z',
            '2025-07-02T10:00:00Z','2025-07-02T10:02:00Z']),
        parent.AIRPORT:['LIRF']*6,
        'RUNWAY_mvt':['25']*6,
        parent.AOBT:['2025-07-01T09:00:00Z']*6,
        parent.TARGET:[100,200,300,400,500,600],
    })
    f=parent.build_features(d)
    parent.add_past_only_flow(f)
    add_strict_history(f)
    assert f['q_runway_past_5m'].to_numpy().tolist()==[0.,1.,1.,2.,0.,1.], f['q_runway_past_5m'].tolist()
    assert f['q_runway_idle_sec'].iloc[4]==7200.,'Cross-day false history'
    assert f['q_runway_idle_sec'].iloc[2]==300.,'Tied event contaminated history'
    assert f['q_runway_idle_sec'].iloc[3]==240.,'Wrong strictly previous timestamp'
    state=fit_or_state(f,np.array([0,1,2,3]))
    q1=queue_matrix(f,np.array([2,3,4]),state)
    assert np.isfinite(q1.to_numpy()).all()
    # Query labels cannot change OR features/capacities.
    d[parent.TARGET]=[1e8]*len(d)
    f2=parent.build_features(d)
    parent.add_past_only_flow(f2)
    add_strict_history(f2)
    state2=fit_or_state(f2,np.array([0,1,2,3]))
    assert q1.equals(queue_matrix(f2,np.array([2,3,4]),state2))
    # Add future movements to a new dataset: previous row counts never change.
    future=pd.concat([d,d.iloc[[0]].assign(**{parent.TAKEOFF:pd.Timestamp('2025-08-01T09:00:00Z')})],ignore_index=True)
    f3=parent.build_features(future)
    parent.add_past_only_flow(f3);add_strict_history(f3)
    for name in ['q_runway_past_5m','q_runway_idle_sec','q_prev_runway_headway_sec']:
        assert np.array_equal(f[name].to_numpy(),f3[name].to_numpy()[:len(f)])
    # Capacity must depend only on FIT, not SELECT/OUTER events.
    fc=f.copy(); fc.loc[4,'runway_past_15m']=1e6
    assert fit_or_state(fc,np.array([0,1,2,3]))==state
    print('SELF-TEST PASS C23: strict past runway/airport events, ties, day boundaries, '
          'future event invariance, FIT-only service capacity, label invariance')


def main():
    c22,parent,c22_sha,c21_sha=load_deps()
    # Preserve C22 CLI, add C23 default output directory only when omitted.
    if '--out-dir' not in sys.argv and not any(x.startswith('--out-dir=') for x in sys.argv):
        sys.argv.extend(['--out-dir','campaign23_research'])
    args=c22.args_from_cli()
    self_test(parent,c22)
    if args.self_test: return 0
    print('='*78)
    print('CAMPAIGN 23 -- QUEUEING THEORY + MACHINE LEARNING / 2025 RESEARCH ONLY')
    print('NO 2026 RANKING READ, NO FINAL REFIT, NO SUBMISSION, NO 2026 PREDICTIONS')
    print('HISTORICALLY EXPLORED OUTERS, NOT INDEPENDENT HOLDOUTS')
    print('='*78,flush=True)
    start=time.perf_counter()
    raw,source=parent.load_training_files(args.data_dir)
    feat=parent.build_features(raw)
    y=parent.make_y(raw)
    assert (feat['_timestamp'].dt.year==2025).all()
    parent.add_past_only_flow(feat)
    print('Building strict past-takeoff headways and recent service history',flush=True)
    add_strict_history(feat)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    parent.write_json(args.out_dir/'manifest.json',{
        'campaign':23,'code_sha256':parent.sha256_file(Path(__file__)),
        'parent22_sha256':c22_sha,'parent21_sha256':c21_sha,
        'inputs':source,
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'queue_note':'Service capacity inferred from FIT-only recent MVT counts; '
                     'M/D/1 wait is a proxy, not a measured queue.',
        'research_only':True,'no_2026':True})
    locked=[]; select_rows=[]; outer_rows=[]; diag=[]; predictions=[]
    for outer in args.outer_months:
        print(f'\n==== OUTER {outer} / SELECT {parent.months_before(outer)} ====',flush=True)
        cuts=parent.select_indices(feat,y,outer,args.embargo_days)
        print('ROWS',{k:len(v) for k,v in cuts.items()},flush=True)
        print('  FIT strictly before SELECT',flush=True)
        pack=fit_models(feat,y,cuts['fit_select'],args,c22,parent)
        prev_name,prev_s,ref_info=c22_locked_candidates(
            pack,feat,y,cuts['select'],args,c22,parent,outer)
        q_s,qs=queue_candidates(pack,feat,cuts['select'],prev_s,args,parent)
        challenger,rows=select_new(feat,y,cuts['select'],prev_s,q_s,args,parent,outer)
        choice=challenger or prev_name
        select_rows.extend([{**r,'stage':'C20'} for r in ref_info['c20rows']]+
                           [{**r,'stage':'C21'} for r in ref_info['c21rows']]+
                           [{**r,'stage':'C22'} for r in ref_info['c22rows']]+
                           [{**r,'stage':'C23','C22_reference':prev_name} for r in rows])
        print(f'  LOCKED SELECT: C22={prev_name} => C23={choice}; '
              f'queue rho={qs["mean_runway_rho"]:.3f}',flush=True)
        del pack,q_s,prev_s
        print('  REFIT strictly before OUTER -- research evaluation only',flush=True)
        pack=fit_models(feat,y,cuts['fit_outer'],args,c22,parent)
        prev=c22_locked_predict(pack,feat,y,cuts['outer'],args,parent,ref_info)
        q_o,qo=queue_candidates(pack,feat,cuts['outer'],prev,args,parent)
        chosen=q_o[challenger] if challenger else prev
        yy=y[cuts['outer']]
        s_ref=parent.score(yy,prev);s_new=parent.score(yy,chosen)
        print(f'  OUTER C22={s_ref["rmse"]:.3f} C23={s_new["rmse"]:.3f} '
              f'[chosen={choice}; LIRF bad={qo["query_LIRF_bad"]}]',flush=True)
        locked.append({'outer':outer,'select':parent.months_before(outer),
                       'n':len(yy),'C22_locked':prev_name,'C23_locked':choice,
                       'C23_challenger':challenger or '',
                       'C22_rmse':s_ref['rmse'],'C23_rmse':s_new['rmse'],
                       **pack['info'],**qo})
        days=feat['_day'].iloc[cuts['outer']].to_numpy()
        for j,(name,pred) in enumerate([('C22_locked',prev),('C23_locked',chosen)]+list(q_o.items())):
            boot=({} if name=='C22_locked' else parent.paired_bootstrap(
                yy,pred,prev,days,args.bootstrap,args.seed+15307+j*397))
            outer_rows.append({'outer':outer,'candidate':name,'selected':name=='C23_locked',
                               **parent.score(yy,pred),**boot})
        ap=feat['airport'].iloc[cuts['outer']].astype(str).to_numpy()
        ok=parent._plausible_proxy(feat,cuts['outer'])
        high=queue_matrix(feat,cuts['outer'],pack['state'])['q_runway_utilization'].to_numpy()>=.80
        base_sq=(yy-prev)**2;new_sq=(yy-chosen)**2
        top=base_sq>=np.quantile(base_sq,.999)
        for label,mask in {'all':np.ones(len(yy),bool),'LIRF_bad':(ap=='LIRF')&~ok,
                           'LIRF_good':(ap=='LIRF')&ok,'LTFM':ap=='LTFM',
                           'runway_util_high':high,'runway_util_low':~high,
                           'ref_top0p1pct':top,'ref_other99p9pct':~top}.items():
            if mask.any():
                diag.append({'outer':outer,'slice':label,'n':int(mask.sum()),
                             'C22_rmse':float(np.sqrt(base_sq[mask].mean())),
                             'C23_rmse':float(np.sqrt(new_sq[mask].mean())),
                             'delta_sse':float((new_sq[mask]-base_sq[mask]).sum()),
                             'C22_sse_fraction':float(base_sq[mask].sum()/base_sq.sum())})
        if args.save_outer_predictions:
            ix=cuts['outer']
            df=pd.DataFrame({'outer':outer,'utc_mvt':feat['_timestamp'].iloc[ix].to_numpy(),
                             'airport':ap,'proxy_ok':ok,'y_true':yy,
                             'pred_C22_locked':prev,'pred_C23_locked':chosen})
            if parent.ID in raw.columns:
                df.insert(0,parent.ID,raw[parent.ID].iloc[ix].to_numpy())
            for name,pred in q_o.items(): df['pred_'+name]=pred
            predictions.append(df)
        del pack,q_o
    for filename,rows in [('locked_choices.csv',locked),
                          ('select_candidates.csv',select_rows),
                          ('outer_candidates_DESCRIPTIVE.csv',outer_rows),
                          ('queue_and_tail_diagnostics.csv',diag)]:
        pd.DataFrame(rows).to_csv(args.out_dir/filename,index=False)
    if predictions:
        combined=pd.concat(predictions,ignore_index=True)
        try:
            combined.to_parquet(args.out_dir/'outer_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:
            combined.to_csv(args.out_dir/'outer_predictions_2025_RESEARCH_ONLY.csv',index=False)
    if args.february_stress:
        feb=(feat['_month'].to_numpy()=='2025-02') & np.isfinite(y)&(y>=0)
        airport=feat['airport'].astype(str).to_numpy()
        descriptions=[]
        for a in sorted(np.unique(airport[feb])):
            m=feb&(airport==a)
            descriptions.append({'airport':a,'n':int(m.sum()),
                                 'n_gt_3600':int((y[m]>3600).sum()),
                                 'p99_target':float(np.quantile(y[m],.99))})
        pd.DataFrame(descriptions).to_csv(args.out_dir/'february_2025_DESCRIPTIVE.csv',index=False)
    n=sum(r['n'] for r in locked)
    pooled={name:float(np.sqrt(sum(r['n']*r[name+'_rmse']**2 for r in locked)/n))
            for name in ('C22','C23')}
    parent.write_json(args.out_dir/'summary.json',{
        'campaign':23,'pooled_rmse':pooled,'locked':locked,
        'seconds':round(time.perf_counter()-start,2),
        'note':'2025 historically explored; no independent holdout; no 2026 data',
        'research_only':True,'no_2026':True})
    print('POOLED MATCHED 2025 OUTERS:',pooled,flush=True)
    print('Research output:',args.out_dir.resolve(),flush=True)
    print('NO FINAL REFIT / NO SUBMISSION / NO 2026 RANKING PREDICTIONS',flush=True)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except (AssertionError,ValueError,KeyError,FileNotFoundError,RuntimeError) as e:
        print(f'FATAL: {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(2)
