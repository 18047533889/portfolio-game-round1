"""Transparent walk-forward engine, preserving failures and trading assumptions."""
from __future__ import annotations
import time
from collections import Counter
from collections.abc import Callable
import numpy as np
import pandas as pd
from portfolio_game.core import check_weights
from research.metrics import performance
from research.models import run_model
from research.data import validate_returns


def holding_returns(returns: np.ndarray, target: np.ndarray, weight_drift: bool) -> tuple:
    """Gross returns, ending pre-trade weights, internal absolute-weight turnover.

    Constant weights imply daily rebalancing inside the holding block. Its turnover
    is computed explicitly for cost sensitivity, rather than charging only monthly.
    """
    r = np.asarray(returns,dtype=float)
    w = np.asarray(target,dtype=float).copy()
    gross = np.empty(len(r)); turnover = np.zeros(len(r))
    for i,row in enumerate(r):
        if not weight_drift:
            if i: turnover[i] = float(np.abs(w-target).sum())
            w = target.copy()
        gross[i] = float(w @ row)
        total = 1.0 + gross[i]
        if total <= 0:
            raise ArithmeticError("Complete capital loss makes subsequent holdings undefined")
        w = w * (1.0 + row) / total
    return gross,w,turnover


def walk_forward(
    frame: pd.DataFrame, config: dict, *, lookback: int=252, rebalance: int=20,
    start: pd.Timestamp | None=None, end: pd.Timestamp | None=None,
    weight_drift: bool=False, cost_bps: float=0.0,
    allocator: Callable=run_model,
) -> dict:
    frame = validate_returns(frame)
    if lookback<2 or rebalance<1 or cost_bps<0:
        raise ValueError("Invalid lookback, rebalance or transaction-cost assumption")
    # Slice end BEFORE any rolling model fit. Start still retains prior training data.
    if end is not None: frame=frame.loc[frame.index<=end]
    first=max(lookback, int(frame.index.searchsorted(start))) if start is not None else lookback
    if first>=len(frame): raise ValueError("No evaluation observations with a complete training window")
    series=pd.Series(np.nan,index=frame.index[first:],name="portfolio_return",dtype=float)
    weights=[]; folds=[]; previous=None; failed=0; degraded=0; statuses=Counter(); durations=[]
    for pos in range(first,len(frame),rebalance):
        train=frame.iloc[pos-lookback:pos]
        test=frame.iloc[pos:min(pos+rebalance,len(frame))]
        fold={"train_start":str(train.index[0].date()),"train_end":str(train.index[-1].date()),
              "test_start":str(test.index[0].date()),"test_end":str(test.index[-1].date()),
              "train_rows":len(train),"test_rows":len(test)}
        begun=time.perf_counter()
        try:
            result=allocator(train.to_numpy(),config)
            w=result["weights"]; check_weights(w,frame.shape[1])
            gross,end_weights,inside_turnover=holding_returns(test.to_numpy(),w,weight_drift)
            rebalance_turnover=float(np.abs(w-(np.zeros_like(w) if previous is None else previous)).sum())
            turnover=inside_turnover.copy();turnover[0]+=rebalance_turnover
            cost=cost_bps/10000.0*turnover
            if np.any(cost>=1): raise ArithmeticError("Transaction-cost assumption consumes all capital")
            net=(1.0-cost)*(1.0+gross)-1.0
            if not np.isfinite(net).all(): raise FloatingPointError("Nonfinite portfolio return")
            series.loc[test.index]=net
            d=result["diagnostics"];status=d["status"];statuses[status]+=1
            is_degraded=(status.startswith(("fallback","emergency")) or d.get("anti_equal_weight_applied",False)
                         or d.get("single_asset_fallback",False))
            degraded+=int(is_degraded)
            fold.update({"success":True,"diagnostics":d,"turnover_l1":float(turnover.sum())})
            weights.append({"date":str(test.index[0].date()),**{str(c):float(v) for c,v in zip(frame.columns,w,strict=True)}})
            previous=end_weights
        except Exception as exc:
            failed+=1; previous=None
            fold.update({"success":False,"error":f"{type(exc).__name__}: {str(exc)[:300]}"})
        elapsed=time.perf_counter()-begun;durations.append(elapsed);fold["seconds"]=elapsed;folds.append(fold)
    summary={"phase_start":str(series.index[0].date()),"phase_end":str(series.index[-1].date()),
        "folds":len(folds),"failed_folds":failed,"failure_rate":failed/len(folds),
        "degraded_folds":degraded,"degradation_rate":degraded/len(folds),"statuses":dict(statuses),
        "fit_and_hold_seconds_median":float(np.median(durations)),
        "fit_and_hold_seconds_p95":float(np.quantile(durations,.95)),
        "performance":performance(series.to_numpy()) if failed==0 else None,
        "assumptions":{"lookback":lookback,"rebalance":rebalance,"weight_drift":weight_drift,
            "cost_bps_per_dollar_traded":cost_bps,"cost_model":"multiplicative_capital_haircut_L1_turnover",
            "teacher_annual_return_definition":"unknown; both CAGR and annualized mean reported"}}
    return {"returns":series,"weights":pd.DataFrame(weights),"folds":folds,"summary":summary}
