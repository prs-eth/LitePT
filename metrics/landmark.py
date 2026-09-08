"""Official 3DTeethLand challenge scoring for landmark predictions.

The metric functions (``voc_ap``, ``voc_ar``, ``eval_det_cls_map``, ``eval_map``,
``score``, ``reformat_scores``) are vendored from the challenge scoring code
(``BracketPrediction/evaluation/metrics.py`` and ``score.py``) so training-time
monitoring matches the official evaluation exactly.

Predictions are exchanged as rows in the challenge ``predictions.csv`` format:
``key, coord_x, coord_y, coord_z, class, score`` where ``key`` is
``{patient_id}_{arch}`` (the scan name stem).
"""

import csv
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CHALLENGE_CLASSES = (
    "Mesial",
    "Distal",
    "InnerPoint",
    "OuterPoint",
    "FacialPoint",
    "Cusp",
)
CSV_FIELDS = ("key", "coord_x", "coord_y", "coord_z", "class", "score")


def voc_ar(dist_thresh_list, recall_values):
    mrec = np.array(dist_thresh_list[::-1])
    mpre = np.array(recall_values[::-1])
    mrec = np.concatenate(([0.0], mrec, [1.0]))
    mpre = np.concatenate(([0.0], mpre, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def eval_det_cls_map(pred, gt, dist_thresh):
    # construct gt objects: {mesh name: {'kp': kp list, 'det': matched list}}
    class_recs = {}
    npos = 0
    for mesh_name in gt.keys():
        keypoints = np.array(gt[mesh_name])
        class_recs[mesh_name] = {"kp": keypoints, "det": [False] * len(keypoints)}
        npos += len(keypoints)
    for mesh_name in pred.keys():
        if mesh_name not in gt:
            class_recs[mesh_name] = {"kp": np.array([]), "det": []}

    # flatten detections and sort by confidence
    mesh_names = []
    confidence = []
    KP = []
    for mesh_name in pred.keys():
        for kp, score in pred[mesh_name]:
            mesh_names.append(mesh_name)
            confidence.append(score)
            KP.append(kp)
    confidence = np.array(confidence)
    KP = np.array(KP)
    sorted_ind = np.argsort(-confidence)
    KP = KP[sorted_ind, ...]
    mesh_names = [mesh_names[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(mesh_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[mesh_names[d]]
        kp = KP[d]
        dmin = np.inf
        KPGT = R["kp"]
        if KPGT.size > 0:
            distance = np.linalg.norm(np.array(kp).reshape(-1, 3) - KPGT, axis=1)
            dmin = min(distance)
            jmin = np.argmin(distance)
        if dmin < dist_thresh:
            if not R["det"][jmin]:
                tp[d] = 1.0
                R["det"][jmin] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap


def eval_det_cls_error(pred, gt, dist_thresh):
    """Same greedy, confidence-sorted matching as eval_det_cls_map, but
    returns the Euclidean distance of each matched (true-positive) pair
    instead of a precision/recall curve.

    Unmatched predictions (false positives) and unmatched GT (misses) are
    excluded -- this is a pure localization-quality metric, kept separate
    from the precision/recall that AP/AR already measure.
    """
    class_recs = {}
    for mesh_name in gt.keys():
        keypoints = np.array(gt[mesh_name])
        class_recs[mesh_name] = {"kp": keypoints, "det": [False] * len(keypoints)}
    for mesh_name in pred.keys():
        if mesh_name not in gt:
            class_recs[mesh_name] = {"kp": np.array([]), "det": []}

    mesh_names = []
    confidence = []
    KP = []
    for mesh_name in pred.keys():
        for kp, score in pred[mesh_name]:
            mesh_names.append(mesh_name)
            confidence.append(score)
            KP.append(kp)
    confidence = np.array(confidence)
    KP = np.array(KP)
    sorted_ind = np.argsort(-confidence)
    KP = KP[sorted_ind, ...]
    mesh_names = [mesh_names[x] for x in sorted_ind]

    distances = []
    for d in range(len(mesh_names)):
        R = class_recs[mesh_names[d]]
        kp = KP[d]
        KPGT = R["kp"]
        if KPGT.size == 0:
            continue
        distance = np.linalg.norm(np.array(kp).reshape(-1, 3) - KPGT, axis=1)
        dmin, jmin = distance.min(), distance.argmin()
        if dmin < dist_thresh and not R["det"][jmin]:
            R["det"][jmin] = True
            distances.append(float(dmin))
    return distances


def error_stats(pred_all_map, gt_all, dist_thresh=3.0):
    """Mean/std Euclidean error (mm) over matched (TP) pairs, pooled and per class.

    Matching mirrors eval_det_cls_map's greedy assignment at a single fixed
    distance threshold (default 3mm, matching LandmarkEvaluator's
    match_distance_threshold). No penalty for unmatched predictions/GT --
    AP/AR already measure precision/recall.
    """
    per_class = {}
    pooled = []
    for cls in CHALLENGE_CLASSES:
        distances = eval_det_cls_error(
            pred_all_map.get(cls, {}), gt_all.get(cls, {}), dist_thresh
        )
        pooled.extend(distances)
        per_class[cls] = {
            "error_mean_mm": float(np.mean(distances)) if distances else None,
            "error_std_mm": float(np.std(distances)) if distances else None,
            "n": len(distances),
        }
    return {
        "error_mean_mm": float(np.mean(pooled)) if pooled else None,
        "error_std_mm": float(np.std(pooled)) if pooled else None,
        "error_n": len(pooled),
        "per_class": per_class,
    }


def eval_map(pred_all, gt_all, dist_thresh=0.1):
    rec, prec, ap = {}, {}, {}
    for classname in gt_all.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls_map(
            pred_all[classname], gt_all[classname], dist_thresh
        )
    return rec, prec, ap


def score(gt_all, pred_all_map):
    """AP/AR per class averaged over 30 distance thresholds (0.0 to 2.9 mm)."""
    ap_per_thresh = {cls: [] for cls in CHALLENGE_CLASSES}
    recall = {cls: [] for cls in CHALLENGE_CLASSES}
    dist_thresh_list = []
    for i in range(0, 30):
        dist_thresh = 0.1 * i
        rec, _, ap = eval_map(pred_all_map, gt_all, dist_thresh=dist_thresh)
        dist_thresh_list.append(dist_thresh)
        for cat in CHALLENGE_CLASSES:
            ap_per_thresh[cat].append(ap[cat])
            # rec is empty when there are no predictions for this class
            recall[cat].append(rec[cat][-1] if len(rec[cat]) else 0.0)

    mean_ap = {cat: float(np.mean(values)) for cat, values in ap_per_thresh.items()}
    mean_ar = {
        cat: float(voc_ar(np.exp(-np.asarray(dist_thresh_list)), recall[cat]))
        for cat in CHALLENGE_CLASSES
    }
    return {"AP": mean_ap, "AR": mean_ar}


def reformat_scores(scores):
    fmt_scores = {}
    for metric in ("AP", "AR"):
        per_class = scores[metric]
        fmt_scores[f"{metric}_cusp"] = per_class["Cusp"]
        fmt_scores[f"{metric}_mesial_distal"] = (per_class["Mesial"] + per_class["Distal"]) / 2
        fmt_scores[f"{metric}_inner_outer"] = (per_class["InnerPoint"] + per_class["OuterPoint"]) / 2
        fmt_scores[f"{metric}_facial"] = per_class["FacialPoint"]
        fmt_scores[f"m{metric}"] = sum(per_class[cls] for cls in CHALLENGE_CLASSES) / len(CHALLENGE_CLASSES)
    return fmt_scores


def project_to_surface(coords, vertices):
    """Snap each predicted landmark to its nearest mesh vertex.

    GT landmarks sit essentially on the surface while predicted offsets
    don't, so this trims the off-surface component of the error.
    """
    if len(coords) == 0:
        return coords
    _, idx = cKDTree(vertices).query(coords)
    return vertices[idx].astype(np.float32)


def prediction_rows(key, pred, class_names):
    """Convert one scan's decoded landmarks to predictions.csv rows."""
    return [
        {
            "key": key,
            "coord_x": float(coord[0]),
            "coord_y": float(coord[1]),
            "coord_z": float(coord[2]),
            "class": class_names[int(cls_idx)],
            "score": float(score),
        }
        for coord, cls_idx, score in zip(pred["coord"], pred["class"], pred["score"])
    ]


def predictions_to_map(rows):
    """Group prediction rows into the {class: {key: [[coord, score]]}} map."""
    pred_map = {cls: {} for cls in CHALLENGE_CLASSES}
    for row in rows:
        if row["class"] not in pred_map:
            continue
        pred_map[row["class"]].setdefault(row["key"], []).append(
            [[row["coord_x"], row["coord_y"], row["coord_z"]], row["score"]]
        )
    return pred_map


def challenge_score(rows, gold, error_dist_thresh=3.0):
    """Score prediction rows against a challenge gold dict.

    Returns the reformatted AP/AR metrics plus pooled and per-class mean/std
    Euclidean error in mm (see error_stats) -- both single call sites
    (LandmarkChallengeScorer, tools/eval_landmarks.py) get the error metric
    for free.
    """
    pred_map = predictions_to_map(rows)
    scores = reformat_scores(score(gold, pred_map))
    error = error_stats(pred_map, gold, dist_thresh=error_dist_thresh)
    scores["error_mean_mm"] = error["error_mean_mm"]
    scores["error_std_mm"] = error["error_std_mm"]
    scores["error_n"] = error["error_n"]
    for cls, stats in error["per_class"].items():
        scores[f"error_mean_mm_{cls}"] = stats["error_mean_mm"]
        scores[f"error_std_mm_{cls}"] = stats["error_std_mm"]
        scores[f"error_n_{cls}"] = stats["n"]
    return scores


def load_gold(path):
    with open(path, "rb") as f:
        gold = pickle.load(f)
    return {cls: dict(gold.get(cls, {})) for cls in CHALLENGE_CLASSES}


def gold_from_dataset(dataset, class_names):
    """Build the challenge gold dict from a landmark dataset's ``__kpt.json`` files.

    Yields the same content as a ``gold_standard.pkl`` collected on the same scans.
    """
    gold = {cls: {} for cls in CHALLENGE_CLASSES}
    for rel_path in dataset.data_list:
        scan_path = os.path.join(dataset.data_root, rel_path)
        coords, classes = dataset._load_landmarks(scan_path)
        key = Path(scan_path).stem
        for coord, cls_idx in zip(coords.tolist(), classes.tolist()):
            gold[class_names[int(cls_idx)]].setdefault(key, []).append(coord)
    return gold


def subset_gold(gold, keys):
    keys = set(keys)
    return {cls: {k: v for k, v in scans.items() if k in keys} for cls, scans in gold.items()}


def write_predictions_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
