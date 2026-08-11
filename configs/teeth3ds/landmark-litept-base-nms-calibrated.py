"""Grid-search winner (base_lc4_np16384) with per-class nms_radius calibrated
on validation (see tools/tune_nms_per_class.py and
exp/landmark/preliminary_nms_grid_search/output.csv). Eval-only: pass
--weight exp/landmark/preliminary_grid_search/base_lc4_np16384/model/model_best.pth
directly to tools/eval_landmarks.py.
"""

_base_ = ["/homes/mlugli/LitePT/exp/landmark/preliminary_grid_search/configs/base_lc4_np16384.py"]

# class order: Mesial, Distal, InnerPoint, OuterPoint, FacialPoint, Cusp
model = dict(nms_radius=[0.75, 0.75, 0.75, 0.75, 0.5, 0.75])
