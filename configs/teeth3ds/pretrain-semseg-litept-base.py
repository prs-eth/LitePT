"""Segmentation pretraining for the landmark detector (LitePT-base).

Same recipe as pretrain-semseg-litept-small.py; the backbone overrides mirror
landmark-litept-base.py exactly so the checkpoint warm-starts that model.

Tuned to warm-start the grid-search winner (base_lc4_np16384): num_points is
16384 to match, batch_size drops to 2 (the setting the winner ran base@16k
with), and epoch drops to 50 -- with 1780 train scans an epoch costs ~7.4x the
landmark run's, so 50 epochs (~13h, still ~3.7x the winner's total gradient
steps) fits the 24h partition limit where 100 would not.
"""

_base_ = ["./pretrain-semseg-litept-small.py"]

save_path = "exp/teeth3ds/pretrain-semseg-litept-base"

batch_size = 2
epoch = 50
num_points = 16384

# the base config's data dicts are baked with its own num_points, so the
# override must be pushed into them explicitly
data = dict(
    train=dict(num_points=num_points),
    val=dict(num_points=num_points),
    test=dict(num_points=num_points),
)

model = dict(
    backbone=dict(
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        dec_channels=(72, 108, 216, 432),
        dec_num_head=(4, 6, 12, 24),
    ),
)
