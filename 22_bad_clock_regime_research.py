#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EUROCONTROL PRC 2026 -- Campaign 22: LIRF bad-source-clock experts.

Research only. Reads ONLY the twelve official labeled training_2025 files.
Does not open, fit or predict on ranking/submitting/2026 files.

Requires the unmodified Campaign 21 file `21_clock_aware_tail_research.py`
in the same directory. The SHA-256 is verified so results remain reproducible.
All new models use FIT labels exclusively. SELECT chooses the winner, and
historically explored 2025 OUTER months are descriptive, not virgin holdouts.

Novel hypothesis: AOBT-improbable LIRF rows account for much of the SSE.
Conditional L2 experts and regularized empirical priors trained on *earlier*
such cases may be preferable to a single mixture for every LIRF movement.
No target clipping. No label-based query gates. No final refit/submission.
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

PARENT_NAME = '21_clock_aware_tail_research.py'
PARENT_SHA = 'f5707cdda883334e9b388c652aee6954f59f1c8d1cd40b94eed1d47b55af4042'
NAMES = (
    'c22_bad_clock_l2_25',
    'c22_bad_clock_l2_50',
    'c22_bad_no_aobt_35',
    'c22_bad_dual_50',
    'c22_bad_prior_20',
    'c22_bad_prior_40',
)
FOLDS = ('2025-03', '2025-06', '2025-07', '2025-10', '2025-12')
EXTRA = ('2025-04', '2025-05', '2025-08', '2025-09', '2025-11')


def load_parent():
    path = Path(__file__).resolve().with_name(PARENT_NAME)
    if not path.is_file():
        raise FileNotFoundError(f'Required script not found beside Campaign 22: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PARENT_SHA:
        raise RuntimeError(f'Campaign 21 dependency differs from tested version; '
                           f'expected SHA256={PARENT_SHA}, actual={digest}. '
                           'Use original Campaign 21 without edits.')
    spec = importlib.util.spec_from_file_location('campaign21_locked', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, digest


def anomaly_classes(feat: pd.DataFrame, idx: np.ndarray, parent) -> np.ndarray:
    """Unambiguous groups derived only from features, never hidden target."""
    gap = feat['aobt_gap_sec'].iloc[idx].to_numpy(float)
    missing = feat['aobt_missing'].iloc[idx].to_numpy(float) > 0
    group = np.full(len(idx), 'other_clock_discrepancy', dtype=object)
    group[gap > 14400] = 'very_old_4h_plus'
    group[gap < 0] = 'after_takeoff'
    group[missing] = 'missing'
    return group


def fit_bad_experts(feat:pd.DataFrame, y:np.ndarray, fit_idx:np.ndarray,
                    args, parent) -> dict:
    # The parent uses a bounded FIT-only sample in smoke mode. Use exactly
    # the same sample for the new experts as the reference, never SELECT labels.
    c21pack = parent.c21_fit(feat, y, fit_idx, args)
    effective = c21pack['parent']['old']['incumbent']['fit_indices']
    airport = feat['airport'].iloc[effective].astype(str).to_numpy()
    lirf = effective[airport == 'LIRF']
    bad = ~parent._plausible_proxy(feat, lirf)
    bad_idx = lirf[bad]
    yy = y[bad_idx]
    dates = feat['_day'].iloc[bad_idx].astype(str).to_numpy()
    qualified = (len(bad_idx) >= args.c22_min_bad_rows and
                 len(np.unique(dates)) >= args.c22_min_bad_days)
    info = {'fit_lirf': int(len(lirf)), 'fit_bad': int(len(bad_idx)),
            'bad_days': int(len(np.unique(dates))), 'active': bool(qualified)}
    learned = {'parent': c21pack, 'bad': None, 'info': info}
    print(f"    C22 LIRF bad-clock: n={len(bad_idx):,}, "
          f"days={len(np.unique(dates)):,}, enabled={qualified}", flush=True)
    if not qualified:
        return learned
    x = parent.matrix(feat, bad_idx, parent.FEATURES)
    x0 = x[parent.FEATURES_NO_AOBT]
    # Both regressors directly minimize squared error, targeting conditional
    # MEAN rather than median. No clipping of training labels or residuals.
    model = parent.fit_experiment_regressor(x, yy, args, tail=True)
    model0 = parent.fit_experiment_regressor(x0, yy, args, tail=True)

    # Empirical-Bayes-like shrinkage of status-group means. Classes are
    # extracted from FIT-only source clocks. Smoothing shrinkage (fixed 120)
    # regularizes sparse classes toward the pooled bad-clock FIT mean.
    global_mean = float(np.mean(yy))
    prior = {}
    classes = anomaly_classes(feat, bad_idx, parent)
    for name in ('missing', 'after_takeoff', 'very_old_4h_plus',
                 'other_clock_discrepancy'):
        mask = classes == name
        n = int(mask.sum())
        avg = float(np.mean(yy[mask])) if n else global_mean
        val = (n*avg + args.c22_prior_smoothing*global_mean) / (
            n + args.c22_prior_smoothing)
        prior[name] = {'count': n, 'mean': float(val)}
    learned['bad'] = {'full': model, 'without_aobt': model0,
                      'prior': prior, 'pool_mean': global_mean}
    return learned


def bad_candidates(pack:dict, feat:pd.DataFrame, idx:np.ndarray,
                   base:np.ndarray, args, parent):
    """Exactly unchanged for all other airports and plausible AOBT rows."""
    arr = np.asarray(base, dtype=float)
    res = {name: arr.copy() for name in NAMES}
    air = feat['airport'].iloc[idx].astype(str).to_numpy()
    bad = (air == 'LIRF') & (~parent._plausible_proxy(feat, idx))
    pos = np.flatnonzero(bad)
    status = {'query_bad': int(len(pos)), 'active': bool(pack['bad'] is not None)}
    if pack['bad'] is None or len(pos) == 0:
        return res, status
    fit = pack['bad']
    x = parent.matrix(feat, idx[pos], parent.FEATURES)
    query_bad_full = np.maximum(0., np.asarray(fit['full'].predict(x), float))
    query_bad_noaobt = np.maximum(0., np.asarray(
        fit['without_aobt'].predict(x[parent.FEATURES_NO_AOBT]), float))
    groups = anomaly_classes(feat, idx[pos], parent)
    prior = np.asarray([fit['prior'][g]['mean'] for g in groups], float)
    old = arr[pos]
    replacements = {
        'c22_bad_clock_l2_25': .75*old + .25*query_bad_full,
        'c22_bad_clock_l2_50': .50*old + .50*query_bad_full,
        'c22_bad_no_aobt_35': .65*old + .35*query_bad_noaobt,
        'c22_bad_dual_50': .50*old + .25*query_bad_full + .25*query_bad_noaobt,
        'c22_bad_prior_20': .80*old + .20*prior,
        'c22_bad_prior_40': .60*old + .40*prior,
    }
    for name,values in replacements.items():
        res[name][pos] = np.maximum(0., values)
    for name,values in res.items():
        parent.invariant(bool(np.all(np.isfinite(values))), 'nonfinite C22 '+name)
        parent.invariant(bool(np.all(values>=0)), 'negative C22 '+name)
        parent.invariant(np.array_equal(values[~bad], arr[~bad]),
                         'C22 changed plausible/non-LIRF row: '+name)
    return res,status


def select_bad(feat, y, indices, reference:np.ndarray, candidates:dict,
               args, parent, outer):
    """Only SELECT labels; 2 chronological halves + paired day bootstrap."""
    yy = y[indices]
    d = feat['_day'].iloc[indices].to_numpy()
    left_idx,right_idx = parent.split_two_halves(feat,indices)
    is_left = np.isin(indices,left_idx)
    ref_left = parent.score(yy[is_left], reference[is_left])['rmse']
    ref_right = parent.score(yy[~is_left], reference[~is_left])['rmse']
    eligible,rows = [],[]
    for j,name in enumerate(NAMES):
        pred = candidates[name]
        b = parent.paired_bootstrap(yy,pred,reference,d,args.bootstrap,
                                     args.seed+99113+j*733)
        first = parent.score(yy[is_left],pred[is_left])['rmse']-ref_left
        second = parent.score(yy[~is_left],pred[~is_left])['rmse']-ref_right
        qualifies = bool(b['delta_rmse'] <= -args.c22_min_gain_sec and
                         max(first,second) <= args.c22_half_regret_sec and
                         b['q95_delta'] < 0.0)
        record={'outer':outer,'select':parent.months_before(outer),
                'candidate':name,'qualified':qualifies,
                'first_delta_vs_C21':float(first),
                'second_delta_vs_C21':float(second),
                **parent.score(yy,pred),**b}
        rows.append(record)
        if qualifies:eligible.append(record)
    selected = (sorted(eligible,key=lambda r:(
        max(r['first_delta_vs_C21'],r['second_delta_vs_C21']),
        r['rmse'],r['candidate']))[0]['candidate'] if eligible else None)
    return selected,rows


def self_test(parent):
    parent.c21_self_test()
    data = pd.DataFrame({
       parent.TAKEOFF:pd.to_datetime([
           '2025-07-01T10:00:00Z','2025-07-01T12:00:00Z',
           '2025-07-01T14:00:00Z','2025-07-01T16:00:00Z']),
       parent.AOBT:[None, '2025-07-01T11:20:00Z',
                    '2025-07-01T06:00:00Z','2025-07-01T15:20:00Z'],
       parent.AIRPORT:['LIRF','LIRF','LIRF','LEBL'],
       parent.TARGET:[1000,400,6000,700]})
    ft=parent.build_features(data)
    idx=np.arange(len(data),dtype=int)
    bad=(ft['airport'].to_numpy()=='LIRF') & ~parent._plausible_proxy(ft,idx)
    parent.invariant(np.array_equal(bad,[True,False,True,False]),
                     'C22 bad-clock gate wrong')
    labels=anomaly_classes(ft,idx,parent)
    parent.invariant(labels.tolist()==['missing','other_clock_discrepancy',
                                      'very_old_4h_plus','other_clock_discrepancy'],
                     'C22 status classes wrong')
    old=np.asarray([400., 200., 600., 300.])
    unchanged,status=bad_candidates({'bad':None},ft,idx,old,None,parent)
    parent.invariant(all(np.array_equal(a,old) for a in unchanged.values()),
                     'Disabled C22 changed incumbent')
    data[parent.TARGET] = [1e6,1e6,1e6,1e6]
    altered=parent.build_features(data)
    parent.invariant(ft[parent.FEATURES].equals(altered[parent.FEATURES]),
                     'C22 gate depends on target labels')
    parent.invariant('2026' not in FOLDS and len(NAMES)==len(set(NAMES)),
                     '2026 scope or duplicate models')
    print('SELF-TEST PASS C22: source-clock regimes, non-LIRF isolation, '
          'target invariance, neutral fallback, 2025-only folds',flush=True)


def args_from_cli():
    p=argparse.ArgumentParser(description='Campaign 22 bad-source-clock specialist research -- 2025 only')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign22_research'))
    p.add_argument('--preset',choices=['smoke','full'],default='smoke')
    p.add_argument('--outer-months',nargs='+',default=list(FOLDS))
    p.add_argument('--extra-months',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--embargo-days',type=int,default=1)
    p.add_argument('--prior-lag-days',type=int,default=1)
    p.add_argument('--tail-threshold',type=float,default=1800.0)
    p.add_argument('--local-tail-quantile',type=float,default=.9)
    p.add_argument('--super-tail-threshold',type=float,default=3600.0)
    p.add_argument('--min-super-tail-train',type=int,default=100)
    p.add_argument('--min-tail-train',type=int,default=300)
    p.add_argument('--min-local-tail-train',type=int,default=120)
    p.add_argument('--min-residual-train',type=int,default=1500)
    p.add_argument('--min-special-train',type=int,default=2500)
    p.add_argument('--min-local-train',type=int,default=4000)
    p.add_argument('--min-gain-sec',type=float,default=1.)
    p.add_argument('--min-incremental-gain-sec',type=float,default=1.)
    p.add_argument('--c21-min-gain-sec',type=float,default=1.)
    p.add_argument('--max-half-regret-sec',type=float,default=1.)
    p.add_argument('--c21-medium-threshold',type=float,default=2400.)
    p.add_argument('--c21-min-medium-events',type=int,default=100)
    p.add_argument('--c22-min-bad-rows',type=int,default=140)
    p.add_argument('--c22-min-bad-days',type=int,default=20)
    p.add_argument('--c22-prior-smoothing',type=float,default=120.)
    p.add_argument('--c22-min-gain-sec',type=float,default=1.)
    p.add_argument('--c22-half-regret-sec',type=float,default=1.)
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--february-stress',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.extra_months: args.outer_months=sorted(set(args.outer_months)|set(EXTRA))
    allowed=set(FOLDS)|set(EXTRA)
    if not args.outer_months or any(m not in allowed for m in args.outer_months):
        p.error('OUTER months must be 2025-03 through 2025-12 (research 2025 only)')
    if args.threads<1 or args.bootstrap<100 or args.embargo_days<0 or args.prior_lag_days<1:
        p.error('Invalid compute/bootstrap/embargo settings')
    if not (0<args.c22_prior_smoothing and args.c22_min_bad_rows>=20 and
            args.c22_min_bad_days>=2 and args.c22_min_gain_sec>=0 and
            args.c22_half_regret_sec>=0):
        p.error('Invalid C22 support/selection settings')
    if not (args.tail_threshold<args.c21_medium_threshold<args.super_tail_threshold):
        p.error('Tail thresholds must be ordered 1800 < 2400 < 3600')
    args.modules=list(('tail','flow','route','moe'))
    args.trees=220 if args.preset=='full' else 65
    args.experiment_trees=180 if args.preset=='full' else 55
    args.max_train_rows=None if args.preset=='full' else 100000
    if args.preset=='smoke':
        args.min_local_train=700
        args.min_special_train=700
        args.min_tail_train=50
        args.min_local_tail_train=30
        args.min_residual_train=250
        args.min_super_tail_train=15
        args.c21_min_medium_events=15
    return args


def diagnostics(feat,y,indices,baseline,selected,parent,outer):
    airport=feat['airport'].iloc[indices].astype(str).to_numpy()
    proxy_ok=parent._plausible_proxy(feat,indices)
    truth=y[indices]
    sq0=(baseline-truth)**2
    sq1=(selected-truth)**2
    base_top=sq0>=np.quantile(sq0,.999)
    for name,mask in {
        'all':np.ones(len(indices),bool),
        'LIRF_bad':(airport=='LIRF')&(~proxy_ok),
        'LIRF_good':(airport=='LIRF')&proxy_ok,
        'LTFM':airport=='LTFM',
        'reference_top_0p1pct':base_top,
        'reference_other_99p9pct':~base_top,
        'LIRF_bad_y_gt_3600':(airport=='LIRF')&(~proxy_ok)&(truth>3600),
    }.items():
        if mask.any():
            yield {'outer':outer,'slice':name,'n':int(mask.sum()),
                   'ref_rmse':float(np.sqrt(sq0[mask].mean())),
                   'C22_rmse':float(np.sqrt(sq1[mask].mean())),
                   'delta_sse':float((sq1[mask]-sq0[mask]).sum()),
                   'ref_sse_share':float(sq0[mask].sum()/sq0.sum())}


def main():
    parent,parent_sha=load_parent()
    args=args_from_cli()
    self_test(parent)
    if args.self_test:return 0
    print('='*78,flush=True)
    print('CAMPAIGN 22: BAD AOBT LIRF SPECIALISTS -- 2025 ONLY / RESEARCH',flush=True)
    print('NO RANKING/SUBMITTING READ; NO FINAL REFIT; NO 2026 PREDICTIONS',flush=True)
    print('ALL OUTER MONTHS HISTORICALLY EXPLORED, NOT INDEPENDENT HOLDOUTS',flush=True)
    print('='*78,flush=True)
    tic=time.perf_counter()
    raw,files=parent.load_training_files(args.data_dir)
    feat=parent.build_features(raw)
    y=parent.make_y(raw)
    parent.invariant(bool((feat['_timestamp'].dt.year==2025).all()),'Non-2025 clock')
    parent.add_past_only_flow(feat)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    parent.write_json(args.out_dir/'manifest.json', {
        'campaign':22,'code_sha256':parent.sha256_file(Path(__file__)),
        'required_parent_sha256':parent_sha,'inputs':files,
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'research_only':True,'no_2026':True,
        'notes':['FIT-only bad-AOBT labels',
                 'Only C21 SELECT-locked model is the comparator',
                 'SELECT halves and paired day bootstrap; OUTER descriptive only']})
    decisions=[]; select_rows=[]; outer_rows=[]; detailed=[]; predictions=[]
    for outer in args.outer_months:
        print(f'\n==== OUTER {outer} / SELECT {parent.months_before(outer)} ====',flush=True)
        cuts=parent.select_indices(feat,y,outer,args.embargo_days)
        print('ROWS',{k:len(v) for k,v in cuts.items()},flush=True)
        print('  FIT before SELECT',flush=True)
        pack=fit_bad_experts(feat,y,cuts['fit_select'],args,parent)
        c20s=parent.predict_campaign20(pack['parent']['parent'],feat,y,cuts['select'],args)
        c20_name,c19_name,c20rows=parent.select_campaign20(
            feat,y,cuts['select'],c20s,args,outer)
        c21s=parent.c21_candidate_predictions(pack['parent'],feat,y,cuts['select'],args,
                                              c20_name,c20s)
        c21_name,c21rows=parent.c21_select(feat,y,cuts['select'],c20_name,c21s,
                                          c20s,args,outer)
        base_s=(c20s[c21_name] if c21_name in c20s else c21s[c21_name])
        bads,info_s=bad_candidates(pack,feat,cuts['select'],base_s,args,parent)
        new_name,rows=select_bad(feat,y,cuts['select'],base_s,bads,args,parent,outer)
        chosen_name=new_name or c21_name
        select_rows.extend([{**r,'stage':'C19_C20'} for r in c20rows]+
                           [{**r,'stage':'C21'} for r in c21rows]+
                           [{**r,'stage':'C22','C21_reference':c21_name} for r in rows])
        print(f'  SELECT C19={c19_name} C20={c20_name} C21={c21_name} '
              f'C22={chosen_name}; bad_query={info_s["query_bad"]}',flush=True)
        del pack,c20s,c21s,bads
        print('  FIT before OUTER / descriptive evaluation only',flush=True)
        pack=fit_bad_experts(feat,y,cuts['fit_outer'],args,parent)
        c20o=parent.predict_campaign20(pack['parent']['parent'],feat,y,cuts['outer'],args)
        c21o=parent.c21_candidate_predictions(pack['parent'],feat,y,cuts['outer'],args,
                                              c20_name,c20o)
        orig=c20o[c21_name] if c21_name in c20o else c21o[c21_name]
        bad_o,info_o=bad_candidates(pack,feat,cuts['outer'],orig,args,parent)
        chosen=bad_o[new_name] if new_name else orig
        yy=y[cuts['outer']]; days=feat['_day'].iloc[cuts['outer']].to_numpy()
        metric1=parent.score(yy,orig); metric2=parent.score(yy,chosen)
        print(f'  OUTER C21={metric1["rmse"]:.3f} C22={metric2["rmse"]:.3f} '
              f'[chosen={chosen_name}; bad={info_o["query_bad"]}]',flush=True)
        decisions.append({'outer':outer,'select':parent.months_before(outer),
                          'n':len(yy),'C20_locked':c20_name,'C21_locked':c21_name,
                          'C22_locked':chosen_name,'C22_challenger':new_name or '',
                          'C21_rmse':metric1['rmse'],'C22_rmse':metric2['rmse'],
                          **{f'fit_{k}':v for k,v in pack['info'].items()}})
        for j,(name,pred) in enumerate([('C21_locked',orig),('C22_locked',chosen)]+
                                       list(bad_o.items())):
            bs=({} if name=='C21_locked' else parent.paired_bootstrap(
                 yy,pred,orig,days,args.bootstrap,args.seed+99713+j*409))
            outer_rows.append({'outer':outer,'candidate':name,'selected':name=='C22_locked',
                               **parent.score(yy,pred),**bs})
        detailed.extend(diagnostics(feat,y,cuts['outer'],orig,chosen,parent,outer))
        if args.save_outer_predictions:
            ix=cuts['outer']; aps=feat['airport'].iloc[ix].astype(str).to_numpy()
            good=parent._plausible_proxy(feat,ix)
            frame=pd.DataFrame({'outer':outer,'utc_mvt':feat['_timestamp'].iloc[ix].to_numpy(),
                                'airport':aps,'proxy_ok':good,'y_true':yy,
                                'pred_C21_locked':orig,'pred_C22_locked':chosen})
            if parent.ID in raw.columns:
                frame.insert(0,parent.ID,raw[parent.ID].iloc[ix].to_numpy())
            for name,pred in bad_o.items():frame['pred_'+name]=pred
            predictions.append(frame)
        del pack,c20o,c21o,bad_o
    for name,rows in [('locked_choices.csv',decisions),
                       ('select_candidates.csv',select_rows),
                       ('outer_candidates_DESCRIPTIVE.csv',outer_rows),
                       ('bad_clock_diagnostics.csv',detailed)]:
        pd.DataFrame(rows).to_csv(args.out_dir/name,index=False)
    if predictions:
        combined=pd.concat(predictions,ignore_index=True)
        try: combined.to_parquet(args.out_dir/'outer_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:
            combined.to_csv(args.out_dir/'outer_predictions_2025_RESEARCH_ONLY.csv',index=False)
    if args.february_stress:
        # Historical descriptive audit only; never used to choose a candidate
        # for 2025-03 or to fit predictors on February labels.
        feb=(feat['_month'].to_numpy()=='2025-02') & np.isfinite(y) & (y>=0)
        apt=feat['airport'].to_numpy()
        stats=[]
        for name in sorted(np.unique(apt[feb])):
            row=feb & (apt==name)
            vals=y[row]
            stats.append({'airport':name,'n':int(len(vals)),
                          'n_gt_3600':int((vals>3600).sum()),
                          'p99_target':float(np.quantile(vals,.99))})
        pd.DataFrame(stats).to_csv(args.out_dir/'february_2025_DESCRIPTIVE.csv',index=False)
    n_total=sum(r['n'] for r in decisions)
    pooled={key:float(np.sqrt(sum(r['n']*r[key+'_rmse']**2 for r in decisions)/n_total))
            for key in ('C21','C22')}
    parent.write_json(args.out_dir/'summary.json', {
        'campaign':22,'pooled_rmse':pooled,'locked':decisions,
        'seconds':round(time.perf_counter()-tic,2),
        'research_only':True,'no_2026':True,
        'note':'Historical exploration; no independent holdout, no final prediction.'})
    print('POOLED 2025 HISTORICAL OUTERS:',pooled,flush=True)
    print('Research outputs:',args.out_dir.resolve(),flush=True)
    print('NO FINAL REFIT / NO SUBMISSION / NO 2026 PREDICTIONS',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssertionError,FileNotFoundError,RuntimeError,ValueError,KeyError) as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
