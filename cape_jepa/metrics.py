
"""Label 1 = bona fide/live; score is live probability. Threshold rule: >= is live."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer_threshold(labels, scores, weights=None):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    fpr, tpr, th = roc_curve(y, s, sample_weight=weights, drop_intermediate=False)
    fnr = 1 - tpr

    # sklearn uses +inf as its initial threshold. Keep all-spoof option finite.
    th = np.where(np.isfinite(th), th, np.nextafter(s.max(), np.inf))
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(th[i]), float((fpr[i] + fnr[i]) * .5 * 100)


def metrics(labels, scores, threshold):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    apcer = float((s[y == 0] >= threshold).mean() * 100)
    bpcer = float((s[y == 1] < threshold).mean() * 100)
    _, eer = eer_threshold(y, s)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)

    return {
        "videos": len(y),
        "live_videos": int(y.sum()),
        "spoof_videos": int((y == 0).sum()),
        "threshold": float(threshold),
        "APCER": apcer,
        "BPCER": bpcer,
        "HTER": (apcer + bpcer) / 2,
        "AUC": float(roc_auc_score(y, s) * 100),
        "EER": eer,
        "TPR_at_FPR_5pct": float(tpr[fpr <= .05].max() * 100)
    }


def youden(y, s):
    fpr, tpr, thresholds = roc_curve(y, s)
    return float(thresholds[np.argmax(tpr - fpr)])
