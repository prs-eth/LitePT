"""LitePT-large landmark detection: small config with the large backbone.

Encoder widths/depths follow configs/structured3d/semseg-litept-large-v1m1.py;
the decoder keeps the small landmark config's block layout (depth 2 per stage,
conv/attn pattern) with large channel widths. Learning rates follow the large
semseg recipe (0.003 base / 0.0003 for blocks).
"""

_base_ = ["./landmark-litept-small.py"]

save_path = "exp/teeth3ds/landmark-litept-large"

model = dict(
    backbone=dict(
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(72, 144, 288, 576, 864),
        enc_num_head=(4, 8, 16, 32, 48),
        dec_channels=(72, 144, 288, 576),
        dec_num_head=(4, 8, 16, 32),
    ),
)

optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(max_lr=[0.003, 0.0003])
param_dicts = [dict(keyword="block", lr=0.0003)]
