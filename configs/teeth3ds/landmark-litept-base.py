"""LitePT-base landmark detection: small config with the base backbone.

Encoder widths/depths follow configs/structured3d/semseg-litept-base-v1m1.py;
the decoder keeps the small landmark config's block layout (depth 2 per stage,
conv/attn pattern) with base channel widths.
"""

_base_ = ["./landmark-litept-small.py"]

save_path = "exp/teeth3ds/landmark-litept-base"

model = dict(
    backbone=dict(
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        dec_channels=(72, 108, 216, 432),
        dec_num_head=(4, 6, 12, 24),
    ),
)
