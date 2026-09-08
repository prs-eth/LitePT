"""Segmentation pretraining for the landmark detector (LitePT-large).

Same recipe as pretrain-semseg-litept-small.py; the backbone overrides mirror
landmark-litept-large.py exactly so the checkpoint warm-starts that model,
including the large learning-rate recipe (0.003 base / 0.0003 for blocks).
Remember to keep num_points equal to the landmark run being warm-started.
"""

_base_ = ["./pretrain-semseg-litept-small.py"]

save_path = "exp/teeth3ds/pretrain-semseg-litept-large"

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
