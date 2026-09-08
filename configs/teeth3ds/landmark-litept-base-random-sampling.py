"""landmark-litept-base-nms-calibrated.py with uniform random vertex sampling
instead of FPS. FPS spreads points evenly over surface area; random sampling
follows the scanner's own triangulation density instead, which is typically
denser around teeth than on the gum/base -- worth trying as an input point
cloud that's naturally weighted toward the regions that matter for landmarks.
Requires retraining (sampling affects the training distribution, not just
inference).
"""

_base_ = ["./landmark-litept-base-nms-calibrated.py"]

save_path = "exp/teeth3ds/landmark-litept-base-random-sampling"

data = dict(
    train=dict(sampling="random"),
    val=dict(sampling="random"),
    test=dict(sampling="random"),
)
