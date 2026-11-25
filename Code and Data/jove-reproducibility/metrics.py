"""Statistical definitions used by the revised manuscript (no fitted classifiers)."""
import math
import numpy as np
import pandas as pd

LOGIT_EPS = 1e-7
LOGLOSS_EPS = 1e-9


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1 + ez)
    return out


def logit(p):
    p = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def discrimination(y, p):
    """Tie-aware ROC AUC and grouped-threshold average precision.

    AP = sum(delta recall * precision) at DISTINCT descending thresholds.
    Single-class AUC and AP with no events are returned as NaN.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    positives = y.sum()
    if positives == 0:
        return float('nan'), float('nan')
    order = np.argsort(-p, kind='stable')
    yy, pp = y[order], p[order]
    ends = np.r_[np.flatnonzero(pp[:-1] != pp[1:]), len(p) - 1]
    tp = np.cumsum(yy)[ends].astype(float)
    fp = ends + 1 - tp
    recall = tp / positives
    ap = np.sum(np.diff(np.r_[0., recall]) * (tp / (ends + 1)))
    if positives == len(y):
        return float('nan'), float(ap)
    fpr = np.r_[0., fp / (len(y) - positives)]
    tpr = np.r_[0., recall]
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    return float(auc), float(ap)


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    auc, ap = discrimination(y, p)
    pi = float(y.mean())
    bs = float(np.mean((y - p) ** 2))
    pp = np.clip(p, LOGLOSS_EPS, 1 - LOGLOSS_EPS)
    return dict(auc=auc, ap=ap, ap_skill=(ap - pi) / (1 - pi) if pi < 1 else np.nan,
                brier=bs, brier_skill=1-bs/(pi*(1-pi)) if 0 < pi < 1 else np.nan,
                log_loss=float(-np.mean(y*np.log(pp)+(1-y)*np.log1p(-pp))))


def fit_offset(y, p, tol=1e-12, max_iter=200):
    """Intercept MLE by bounded bisection; report failed/single-class fits."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or not (0 < y.sum() < len(y)):
        return dict(alpha=np.nan, converged=False, iterations=0, residual=np.nan,
                    status='single_class_or_empty_history')
    z, target = logit(p), y.mean()
    lo, hi = -50., 50.
    if not (sigmoid(z+lo).mean() < target < sigmoid(z+hi).mean()):
        return dict(alpha=np.nan, converged=False, iterations=0, residual=np.nan,
                    status='root_not_bracketed')
    for iteration in range(1, max_iter+1):
        alpha = (lo+hi)/2
        residual = float(sigmoid(z+alpha).mean()-target)
        if abs(residual) <= tol:
            return dict(alpha=alpha, converged=True, iterations=iteration,
                        residual=residual, status='ok')
        if residual > 0:
            hi = alpha
        else:
            lo = alpha
    return dict(alpha=np.nan, converged=False, iterations=max_iter,
                residual=residual, status='iteration_limit')


def calibration(y, p):
    """CITL and unpenalized intercept/slope diagnostic with step halving.

    Sparse or separated fits are flagged, not replaced by plausible numbers.
    Wald intervals are descriptive, conditional on this finite-sample fit.
    """
    fit = fit_offset(y, p)
    result = dict(citl=fit['alpha'], citl_status=fit['status'],
                  citl_ci_low=np.nan, citl_ci_high=np.nan,
                  calibration_intercept=np.nan, calibration_slope=np.nan,
                  slope_ci_low=np.nan, slope_ci_high=np.nan,
                  slope_status='not_estimable')
    if not fit['converged']:
        return result
    lp = logit(p)
    mu = sigmoid(lp+fit['alpha'])
    se = 1/np.sqrt(np.sum(mu*(1-mu)))
    result.update(citl_ci_low=fit['alpha']-1.96*se, citl_ci_high=fit['alpha']+1.96*se)
    x = np.column_stack([np.ones(len(lp)), lp])
    beta = np.array([0., 1.])
    converged = False
    for _ in range(200):
        eta = x @ beta
        mu = sigmoid(eta)
        info = x.T @ (np.clip(mu*(1-mu), 1e-12, None)[:, None]*x)
        try:
            step = np.linalg.solve(info, x.T@(y-mu))
        except np.linalg.LinAlgError:
            break
        old_ll = np.sum(y*eta-np.logaddexp(0, eta))
        scale = 1.
        while scale >= 2**-30:
            new_beta = beta+scale*step
            eta_new = x@new_beta
            if np.sum(y*eta_new-np.logaddexp(0, eta_new)) >= old_ll-1e-12:
                beta = new_beta
                break
            scale /= 2
        else:
            break
        if np.max(np.abs(scale*step)) < 1e-8:
            converged = True
            break
    if converged and np.all(np.isfinite(beta)) and np.max(np.abs(beta)) < 100:
        mu = sigmoid(x@beta)
        info = x.T@(np.clip(mu*(1-mu), 1e-12, None)[:, None]*x)
        try:
            covariance = np.linalg.inv(info)
            if np.linalg.cond(info) > 1e12 or np.any(np.diag(covariance) < 0):
                return result
            slope_se = np.sqrt(covariance[1, 1])
            result.update(calibration_intercept=float(beta[0]), calibration_slope=float(beta[1]),
                          slope_ci_low=float(beta[1]-1.96*slope_se),
                          slope_ci_high=float(beta[1]+1.96*slope_se), slope_status='ok')
        except np.linalg.LinAlgError:
            pass
    return result


def reliability_bins(y, p):
    if np.all(np.asarray(p) == p[0]):
        codes = np.zeros(len(p), dtype=int)
    else:
        codes = pd.qcut(pd.Series(p), q=10, duplicates='drop', labels=False).to_numpy()
    grouped = pd.DataFrame(dict(y=y, p=p, bin=codes)).groupby('bin', observed=True)
    rows = []
    for k, g in grouped:
        rows.append(dict(bin=int(k)+1, n=len(g), events=int(g.y.sum()),
                         predicted=float(g.p.mean()), observed=float(g.y.mean()),
                         probability_min=float(g.p.min()), probability_max=float(g.p.max())))
    return rows


def ranked_indices(p, firm_id):
    return np.lexsort((np.asarray(firm_id, dtype=str), -np.asarray(p)))


def capacity(y, p, firm_id, q):
    k = max(1, math.ceil(q*len(y)))
    idx = ranked_indices(p, firm_id)[:k]
    found, total = int(np.asarray(y)[idx].sum()), int(np.sum(y))
    precision = found/k
    return dict(q=q, k=k, captured=found, precision=precision,
                recall=found/total if total else np.nan,
                lift=precision/(total/len(y)) if total else np.nan), idx


def classification(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(np.sum((y==1)&(yhat==1))); fp = int(np.sum((y==0)&(yhat==1)))
    tn = int(np.sum((y==0)&(yhat==0))); fn = int(np.sum((y==1)&(yhat==0)))
    precision = tp/(tp+fp) if tp+fp else np.nan
    recall = tp/(tp+fn) if tp+fn else np.nan
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                predicted_positive_rate=float(np.mean(yhat)))
