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


def challenge_score(rows, gold):
    """Score prediction rows against a challenge gold dict; returns the reformatted metrics."""
    return reformat_scores(score(gold, predictions_to_map(rows)))


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
