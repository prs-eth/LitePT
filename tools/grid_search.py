"""Generate configs and sbatch files for the landmark hyperparameter grid search.

Grid dimensions: model size (small/base/large) x lambda_coord x num_points x
epoch x positive_radius x focal_alpha x focal_gamma, plus normals and/or
project_to_surface when their --sweep-* flag is given. Everything else is
fixed (lambda_focal pinned to 1.0 since only the coord/focal ratio matters
under AdamW).

--tooth-loss, --normals, --project-to-surface, --heat-gaussian, and
--nms-radius apply to every experiment in the grid rather than adding a
grid dimension:
    --tooth-loss           auxiliary tooth-index classification loss
    --normals              per-vertex surface normals as an extra input
                            feature (backbone in_channels 3 -> 6), fixed on
                            for the whole grid (mutually exclusive with
                            --sweep-normals)
    --project-to-surface   snap predicted landmarks to the nearest mesh
                            vertex in the val/test eval loops, fixed on for
                            the whole grid (mutually exclusive with
                            --sweep-project-to-surface)
    --heat-gaussian         Gaussian-smoothed heat target (by distance to
                            the matched landmark) instead of a hard 0/1
                            target within positive_radius, fixed on for the
                            whole grid (mutually exclusive with
                            --sweep-heat-gaussian). Default is the hard
                            0/1 target
    --nms-radius            one value for all classes (default), or 6 --
                            one per class in CLASS_ORDER -- to plug in
                            thresholds already calibrated for a trained
                            model with tools/tune_nms_per_class.py

--epoch, --positive-radius, --focal-alpha, --focal-gamma,
--gaussian-sigma-ratio, --sweep-normals, --sweep-project-to-surface, and
--sweep-heat-gaussian *do* add a grid dimension:
    --epoch                 total training epoch budget, one or more
                            values, crossed with the rest of the grid.
                            Must be divisible by the base config's
                            eval_epoch (10 for all landmark configs) --
                            data.train.loop scales up automatically
    --positive-radius       ground-truth assignment radius in mm, one or
                            more values, crossed with the rest of the grid
    --focal-alpha           focal loss alpha (positive/negative balance),
                            one or more values, crossed with the rest
    --focal-gamma           focal loss gamma (easy-example down-weighting),
                            one or more values, crossed with the rest
    --gaussian-sigma-ratio  sigma as a fraction of positive_radius, one or
                            more values; only takes effect together with
                            --heat-gaussian / --sweep-heat-gaussian
    --sweep-normals         generate both with- and without-normals
                            variants, crossed with the rest of the grid
    --sweep-project-to-surface  generate both with- and without-
                            project-to-surface variants, crossed with the
                            rest of the grid
    --sweep-heat-gaussian   generate both Gaussian and hard-target
                            variants (one config per --gaussian-sigma-ratio
                            value, plus one hard-target baseline), crossed
                            with the rest of the grid

Example:
    python tools/grid_search.py --output-root exp/landmark --name grid_v1
    python tools/grid_search.py --output-root exp/landmark --name grid_v2 \\
        --lambda-coord 4 8 16 --project-to-surface --normals \\
        --nms-radius 0.75 0.75 0.75 0.75 0.5 0.75
    python tools/grid_search.py --output-root exp/landmark --name grid_v3 \\
        --lambda-coord 6 --positive-radius 1.5 2.0 2.5 --sweep-normals \\
        --project-to-surface --nms-radius 0.75 0.75 0.75 0.75 0.5 0.75
    python tools/grid_search.py --output-root exp/landmark --name grid_v4 \\
        --lambda-coord 6 --epoch 300 --positive-radius 1.0 --normals \\
        --project-to-surface --nms-radius 0.75 0.75 0.75 0.75 0.5 0.75

creates:
    exp/landmark/grid_v1/
        configs/<exp>.py      one config per grid point
        sbatch/<exp>.sbatch   one single-GPU sbatch per grid point
        logs/                 slurm stdout/stderr
        submit_all.sh         sbatch everything
        manifest.csv          grid point -> config/sbatch paths
        <exp>/                training outputs (created by the runs, so wandb
                              runs are named "<grid_name>/<exp_name>")

Once the runs are done (or while they are still going), aggregate the results
into <grid_root>/summary.csv, ranked by best validation mAP:
    python tools/grid_search.py --collect exp/landmark/grid_v1

Per experiment the summary reports the epoch with the best val mAP (with all
val challenge metrics and the test mAP/mAR at that same epoch) plus the
last scored epoch, reading the challenge_scores.jsonl files that the
LandmarkChallengeScorer hook writes into each run directory.
"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_SIZES = ("small", "base", "large")
DEFAULT_LAMBDA_COORD = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_NUM_POINTS = (8192, 16384)
DEFAULT_FOLD = (
    "/homes/mlugli/BracketPrediction/Teeth3DS/splits/"
    "3DTeethland_challenge_train_validation_test_split"
)
DEFAULT_EPOCH = (100,)  # matches configs/_base_/default_runtime.py's default
DEFAULT_NMS_RADIUS = (1.0,)
DEFAULT_POSITIVE_RADIUS = (2.0,)  # matches models/landmark.py's LandmarkDetector default
DEFAULT_FOCAL_ALPHA = (0.25,)  # matches models/landmark.py's LandmarkDetector default
DEFAULT_FOCAL_GAMMA = (2.0,)  # matches models/landmark.py's LandmarkDetector default
DEFAULT_GAUSSIAN_SIGMA_RATIO = (1.0 / 3.0,)  # matches models/landmark.py's LandmarkDetector default
# class order data.names / model.class_names are built in (see
# landmark-litept-small.py) -- a 6-value --nms-radius must follow it
CLASS_ORDER = ("Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint", "Cusp")

# fixed for the whole grid
FIXED_MODEL_PARAMS = dict(
    lambda_focal=1.0,
    score_threshold=0.15,
    max_predictions_per_class=64,
)


def _collect_keys(with_tooth):
    keys = [
        "coord",
        "grid_coord",
        "landmark_coord",
        "landmark_class",
        "coord_center",
        "coord_scale",
        "name",
        "full_path",
    ]
    return keys + ["landmark_tooth"] if with_tooth else keys


def _transform_block(collect_keys, train_aug):
    lines = []
    if train_aug:
        lines += [
            '    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", p=0.5),',
            '    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="x", p=0.5),',
            '    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="y", p=0.5),',
        ]
    lines += [
        '    dict(type="NormalizeCoord"),',
        '    dict(type="GridCoord", grid_size=0.01),',
        '    dict(type="ToTensor"),',
        "    dict(",
        '        type="Collect",',
        f"        keys={collect_keys!r},",
        "        offset_keys_dict=offset_keys,",
        "        feat_keys=feat_keys,",
        "    ),",
    ]
    return "[\n" + "\n".join(lines) + "\n]"


def _data_override_block(tooth_loss, normals):
    """feat_keys/collect_keys/transform overrides for --tooth-loss / --normals.

    Both toggles change what Collect keeps per sample, and Collect's
    "keys"/"feat_keys" live inside a transform *list*, which the config
    merger replaces wholesale rather than merging element-wise -- so
    train/val/test transforms must be redefined in full here (mirrors
    configs/teeth3ds/landmark-litept-base-{tooth-loss,normals}.py).

    Segmentation masks (for the tooth loss) are only available for the
    train/val folds -- the test fold never gets landmark_tooth or
    load_segment=True (see landmark-litept-base-tooth-loss.py).
    """
    feat_keys = ["coord", "normal"] if normals else ["coord"]
    train_val_keys = _collect_keys(with_tooth=tooth_loss)
    test_keys = _collect_keys(with_tooth=False)

    lines = [
        f"feat_keys = {feat_keys!r}",
        "",
        'offset_keys = dict(offset="coord", landmark_offset="landmark_coord")',
        "",
        f"_train_transform = {_transform_block(train_val_keys, train_aug=True)}",
        "",
        f"_val_transform = {_transform_block(train_val_keys, train_aug=False)}",
        "",
    ]
    if tooth_loss:
        lines += [f"_test_transform = {_transform_block(test_keys, train_aug=False)}", ""]
        test_transform = "_test_transform"
    else:
        test_transform = "_val_transform"

    lines += [
        "data = dict(",
        f"    train=dict(num_points=num_points, fold=fold, load_segment={tooth_loss}, transform=_train_transform),",
        f"    val=dict(num_points=num_points, fold=fold, load_segment={tooth_loss}, transform=_val_transform),",
        f"    test=dict(num_points=num_points, fold=fold, load_segment=False, transform={test_transform}),",
        ")",
    ]
    return "\n".join(lines)


def _hooks_override_block():
    """Full hooks list with project_to_surface wired into the eval hooks.

    Needs to mirror configs/teeth3ds/landmark-litept-small.py's hooks list
    exactly -- like the data block above, the list is replaced wholesale by
    the config merger, so there's no way to override just one kwarg of one
    hook without restating the rest.
    """
    return "\n".join(
        [
            "hooks = [",
            '    dict(type="CheckpointLoader"),',
            '    dict(type="ModelHook"),',
            '    dict(type="IterationTimer", warmup_iter=2),',
            '    dict(type="InformationWriter"),',
            '    dict(type="LandmarkEvaluator", match_distance_threshold=3.0, project_to_surface=True),',
            '    dict(',
            '        type="LandmarkChallengeScorer",',
            '        splits=("val", "test"),',
            "        gold_path=None,",
            "        project_to_surface=True,",
            "    ),",
            '    dict(type="CheckpointSaver", save_freq=None),',
            '    dict(type="PreciseEvaluator", test_last=False),',
            "]",
        ]
    )


def render_config(
    base_config,
    save_path,
    num_worker,
    num_points,
    fold,
    epoch,
    lambda_coord,
    positive_radius,
    focal_alpha,
    focal_gamma,
    heat_gaussian,
    gaussian_sigma_ratio,
    nms_radius,
    tooth_loss,
    normals,
    project_to_surface,
    **fixed_model_params,
):
    # single value -> uniform across classes; 6 values -> one per class, in
    # CLASS_ORDER (e.g. thresholds calibrated by tools/tune_nms_per_class.py)
    nms_radius_value = nms_radius[0] if len(nms_radius) == 1 else list(nms_radius)
    model_lines = [
        f"    lambda_coord={lambda_coord},",
        f"    lambda_focal={fixed_model_params['lambda_focal']},",
        f"    focal_alpha={focal_alpha},",
        f"    focal_gamma={focal_gamma},",
        f"    heat_target_gaussian={heat_gaussian},",
        f"    gaussian_sigma_ratio={gaussian_sigma_ratio},",
        f"    positive_radius={positive_radius},",
        f"    nms_radius={nms_radius_value!r},",
        f"    score_threshold={fixed_model_params['score_threshold']},",
        f"    max_predictions_per_class={fixed_model_params['max_predictions_per_class']},",
    ]
    if len(nms_radius) > 1:
        model_lines.insert(7, f"    # nms_radius order: {', '.join(CLASS_ORDER)}")
    if tooth_loss:
        model_lines += [
            "    predict_tooth=True,",
            "    num_tooth_classes=17,",
            "    lambda_tooth=1.0,",
        ]
    if normals:
        model_lines.append("    backbone=dict(in_channels=6),")

    parts = [
        '"""Auto-generated by tools/grid_search.py -- do not edit by hand."""',
        "",
        f'_base_ = ["{base_config}"]',
        "",
        f'save_path = "{save_path}"',
        f"num_worker = {num_worker}",
        f"num_points = {num_points}",
        f'fold = "{fold}"',
        # total training epochs; must be divisible by eval_epoch (from the base
        # config) -- data.train.loop = epoch // eval_epoch is computed
        # automatically in engines/defaults.py at launch time
        f"epoch = {epoch}",
        "",
        "model = dict(",
        *model_lines,
        ")",
        "",
    ]
    if tooth_loss or normals:
        parts.append(_data_override_block(tooth_loss, normals))
    else:
        parts += [
            "data = dict(",
            "    train=dict(num_points=num_points, fold=fold),",
            "    val=dict(num_points=num_points, fold=fold),",
            "    test=dict(num_points=num_points, fold=fold),",
            ")",
        ]
    if project_to_surface:
        parts += ["", _hooks_override_block()]
    return "\n".join(parts) + "\n"


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:1
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/{exp_name}_%j.out
#SBATCH --error={log_dir}/{exp_name}_%j.err

cd {repo_root} || exit 1

source {repo_root}/.venv/bin/activate
export PYTHONPATH=./

python tools/train.py \\
  --config-file {config_path} \\
  --num-gpus 1
"""

SUBMIT_ALL_TEMPLATE = """#!/bin/bash
# Submit every grid-search job. Generated by tools/grid_search.py.
cd "$(dirname "$0")" || exit 1
for f in sbatch/*.sbatch; do
    sbatch "$f"
done
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--collect",
        metavar="GRID_ROOT",
        help="aggregate an existing grid's results into <GRID_ROOT>/summary.csv "
        "instead of generating a new grid",
    )
    parser.add_argument(
        "--output-root",
        help="folder the grid directory is created in, e.g. exp/landmark "
        "(required unless --collect is used)",
    )
    parser.add_argument(
        "--name",
        default="landmark_grid",
        help="name of the grid directory (also the wandb run name prefix)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_SIZES), choices=MODEL_SIZES
    )
    parser.add_argument(
        "--lambda-coord", nargs="+", type=float, default=list(DEFAULT_LAMBDA_COORD)
    )
    parser.add_argument(
        "--num-points", nargs="+", type=int, default=list(DEFAULT_NUM_POINTS)
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPOCH),
        help="total training epoch budget -- a real grid dimension like "
        "--lambda-coord, crossed with the rest of the grid. Must be "
        "divisible by the base config's eval_epoch (10 for all landmark "
        f"configs); data.train.loop scales up automatically (default "
        f"{DEFAULT_EPOCH[0]})",
    )
    parser.add_argument(
        "--positive-radius",
        nargs="+",
        type=float,
        default=list(DEFAULT_POSITIVE_RADIUS),
        help="ground-truth assignment radius in mm (model.positive_radius) -- "
        "a real grid dimension like --lambda-coord, crossed with the rest "
        f"of the grid (default {DEFAULT_POSITIVE_RADIUS[0]:g})",
    )
    parser.add_argument(
        "--focal-alpha",
        nargs="+",
        type=float,
        default=list(DEFAULT_FOCAL_ALPHA),
        help="focal loss alpha, positive/negative balance (model.focal_alpha) -- "
        "a real grid dimension like --lambda-coord, crossed with the rest "
        f"of the grid (default {DEFAULT_FOCAL_ALPHA[0]:g})",
    )
    parser.add_argument(
        "--focal-gamma",
        nargs="+",
        type=float,
        default=list(DEFAULT_FOCAL_GAMMA),
        help="focal loss gamma, easy-example down-weighting (model.focal_gamma) "
        "-- a real grid dimension like --lambda-coord, crossed with the rest "
        f"of the grid (default {DEFAULT_FOCAL_GAMMA[0]:g})",
    )
    gaussian_group = parser.add_mutually_exclusive_group()
    gaussian_group.add_argument(
        "--heat-gaussian",
        action="store_true",
        help="use a Gaussian-smoothed heat target (by distance to the matched "
        "landmark) instead of a hard 0/1 target within positive_radius, for "
        "every experiment in the grid (mutually exclusive with "
        "--sweep-heat-gaussian). Default is the hard 0/1 target",
    )
    gaussian_group.add_argument(
        "--sweep-heat-gaussian",
        action="store_true",
        help="generate both Gaussian and hard-target variants, crossed with "
        "the rest of the grid, instead of a single fixed setting",
    )
    parser.add_argument(
        "--gaussian-sigma-ratio",
        nargs="+",
        type=float,
        default=list(DEFAULT_GAUSSIAN_SIGMA_RATIO),
        help="Gaussian sigma as a fraction of positive_radius (model."
        "gaussian_sigma_ratio), only used when --heat-gaussian or "
        "--sweep-heat-gaussian is set -- a real grid dimension like "
        f"--lambda-coord (default {DEFAULT_GAUSSIAN_SIGMA_RATIO[0]:.3g})",
    )
    parser.add_argument("--fold", default=DEFAULT_FOLD)
    parser.add_argument(
        "--nms-radius",
        nargs="+",
        type=float,
        default=list(DEFAULT_NMS_RADIUS),
        help="NMS radius (mm), applied to every experiment in the grid (not a "
        "grid dimension). Pass one value for all classes, or 6 -- one per "
        f"class in order {', '.join(CLASS_ORDER)} -- to plug in thresholds "
        "already calibrated with tools/tune_nms_per_class.py",
    )
    parser.add_argument(
        "--tooth-loss",
        action="store_true",
        help="enable the auxiliary tooth-index classification loss "
        "(predict_tooth=True, lambda_tooth=1.0) for every experiment in the grid",
    )
    normals_group = parser.add_mutually_exclusive_group()
    normals_group.add_argument(
        "--normals",
        action="store_true",
        help="add per-vertex surface normals as an extra input feature "
        "(backbone in_channels 3 -> 6) for every experiment in the grid",
    )
    normals_group.add_argument(
        "--sweep-normals",
        action="store_true",
        help="generate both with- and without-normals variants, crossed with "
        "the rest of the grid, instead of a single fixed setting",
    )
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--project-to-surface",
        action="store_true",
        help="snap predicted landmarks to the nearest mesh vertex in the "
        "val/test eval loops (LandmarkEvaluator, LandmarkChallengeScorer) "
        "for every experiment in the grid",
    )
    project_group.add_argument(
        "--sweep-project-to-surface",
        action="store_true",
        help="generate both with- and without-project-to-surface variants, "
        "crossed with the rest of the grid, instead of a single fixed setting",
    )
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="30GB")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--account", default="grana_maxillo")
    parser.add_argument("--partition", default="all_usr_prod")
    return parser.parse_args()


def read_scores(run_dir):
    """Read a run's challenge_scores.jsonl into {split: {epoch: scores}}.

    Later records win for a repeated (split, epoch), which dedupes evals
    replayed after a resume.
    """
    scores_path = Path(run_dir) / "challenge_scores.jsonl"
    per_split = {}
    if not scores_path.is_file():
        return per_split
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            split, epoch = record.pop("split"), record.pop("epoch")
            per_split.setdefault(split, {})[epoch] = record
    return per_split


def _round_or_none(value):
    """round(value, 4), passing None through -- error_mean_mm/error_std_mm
    can be None when a class has no matched (TP) pairs yet (e.g. early
    epochs), and round(None, 4) raises.
    """
    return round(value, 4) if isinstance(value, (int, float)) else value


def collect(grid_root):
    grid_root = Path(grid_root).resolve()
    manifest_path = grid_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"not a grid directory (no manifest.csv): {grid_root}")
    with open(manifest_path) as f:
        manifest = list(csv.DictReader(f))

    summary_rows = []
    for exp in manifest:
        row = {key: exp[key] for key in ("exp_name", "model", "lambda_coord", "num_points")}
        row.update(
            {
                key: exp.get(key, "")
                for key in (
                    "epoch",
                    "positive_radius",
                    "focal_alpha",
                    "focal_gamma",
                    "heat_gaussian",
                    "gaussian_sigma_ratio",
                    "nms_radius",
                    "tooth_loss",
                    "normals",
                    "project_to_surface",
                )
            }
        )
        per_split = read_scores(exp["save_path"])
        val, test = per_split.get("val", {}), per_split.get("test", {})
        if not val:
            row["status"] = "no_scores"
            summary_rows.append(row)
            continue
        best_epoch = max(val, key=lambda epoch: val[epoch]["mAP"])
        last_epoch = max(val)
        row.update(
            status="ok",
            epochs_scored=len(val),
            best_epoch=best_epoch,
            val_mAP=round(val[best_epoch]["mAP"], 4),
            val_mAR=round(val[best_epoch]["mAR"], 4),
            test_mAP=round(test[best_epoch]["mAP"], 4) if best_epoch in test else None,
            test_mAR=round(test[best_epoch]["mAR"], 4) if best_epoch in test else None,
            **{
                f"val_{key}": _round_or_none(value)
                for key, value in val[best_epoch].items()
                if key not in ("mAP", "mAR")
            },
            **(
                {
                    f"test_{key}": _round_or_none(value)
                    for key, value in test[best_epoch].items()
                    if key not in ("mAP", "mAR")
                }
                if best_epoch in test
                else {}
            ),
            last_epoch=last_epoch,
            val_mAP_last=round(val[last_epoch]["mAP"], 4),
            val_mAR_last=round(val[last_epoch]["mAR"], 4),
            test_mAP_last=round(test[last_epoch]["mAP"], 4) if last_epoch in test else None,
            test_mAR_last=round(test[last_epoch]["mAR"], 4) if last_epoch in test else None,
        )
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: r.get("val_mAP", -1.0), reverse=True)
    fieldnames = max((list(r.keys()) for r in summary_rows), key=len)
    summary_path = grid_root / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    scored = sum(1 for r in summary_rows if r.get("status") == "ok")
    print(f"wrote {summary_path} ({scored}/{len(summary_rows)} experiments with scores)")


def main():
    args = parse_args()

    if args.collect:
        collect(args.collect)
        return
    if not args.output_root:
        raise SystemExit("--output-root is required unless --collect is used")
    if len(args.nms_radius) not in (1, len(CLASS_ORDER)):
        raise SystemExit(
            f"--nms-radius takes 1 value or {len(CLASS_ORDER)} (one per class, "
            f"order {', '.join(CLASS_ORDER)}), got {len(args.nms_radius)}"
        )

    grid_root = (Path(args.output_root) / args.name).resolve()
    config_dir = grid_root / "configs"
    sbatch_dir = grid_root / "sbatch"
    log_dir = grid_root / "logs"
    for d in (config_dir, sbatch_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    fold = Path(args.fold)
    if not fold.is_dir():
        raise FileNotFoundError(f"fold folder not found: {fold}")

    normals_values = (False, True) if args.sweep_normals else (args.normals,)
    project_values = (
        (False, True) if args.sweep_project_to_surface else (args.project_to_surface,)
    )
    # (heat_gaussian, gaussian_sigma_ratio) pairs, not an independent product --
    # gaussian_sigma_ratio is a no-op when heat_gaussian is False, so sweeping
    # it there would just generate identical-behavior duplicate configs
    if args.sweep_heat_gaussian:
        heat_gaussian_combos = [(False, DEFAULT_GAUSSIAN_SIGMA_RATIO[0])] + [
            (True, ratio) for ratio in args.gaussian_sigma_ratio
        ]
    elif args.heat_gaussian:
        heat_gaussian_combos = [(True, ratio) for ratio in args.gaussian_sigma_ratio]
    else:
        heat_gaussian_combos = [(False, DEFAULT_GAUSSIAN_SIGMA_RATIO[0])]

    manifest_rows = []
    for (
        size,
        lambda_coord,
        num_points,
        epoch,
        positive_radius,
        focal_alpha,
        focal_gamma,
        (heat_gaussian, gaussian_sigma_ratio),
        normals,
        project_to_surface,
    ) in product(
        args.models,
        args.lambda_coord,
        args.num_points,
        args.epoch,
        args.positive_radius,
        args.focal_alpha,
        args.focal_gamma,
        heat_gaussian_combos,
        normals_values,
        project_values,
    ):
        base_config = REPO_ROOT / "configs/teeth3ds" / f"landmark-litept-{size}.py"
        if not base_config.is_file():
            raise FileNotFoundError(f"base config not found: {base_config}")

        exp_name = (
            f"{size}_lc{lambda_coord:g}_np{num_points}_ep{epoch}_pr{positive_radius:g}"
            f"_fa{focal_alpha:g}_fg{focal_gamma:g}"
        )
        if heat_gaussian:
            exp_name += f"_gauss{gaussian_sigma_ratio:g}"
        if args.tooth_loss:
            exp_name += "_tooth"
        if normals:
            exp_name += "_normals"
        if project_to_surface:
            exp_name += "_proj"
        if len(args.nms_radius) > 1:
            exp_name += "_nmscal"
        config_path = config_dir / f"{exp_name}.py"
        sbatch_path = sbatch_dir / f"{exp_name}.sbatch"

        config_path.write_text(
            render_config(
                base_config=base_config,
                save_path=grid_root / exp_name,
                num_worker=args.num_worker,
                num_points=num_points,
                fold=fold,
                epoch=epoch,
                lambda_coord=lambda_coord,
                positive_radius=positive_radius,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                heat_gaussian=heat_gaussian,
                gaussian_sigma_ratio=gaussian_sigma_ratio,
                nms_radius=args.nms_radius,
                tooth_loss=args.tooth_loss,
                normals=normals,
                project_to_surface=project_to_surface,
                **FIXED_MODEL_PARAMS,
            )
        )
        sbatch_path.write_text(
            SBATCH_TEMPLATE.format(
                exp_name=exp_name,
                cpus=args.cpus,
                account=args.account,
                partition=args.partition,
                time=args.time,
                mem=args.mem,
                log_dir=log_dir,
                repo_root=REPO_ROOT,
                config_path=config_path,
            )
        )
        manifest_rows.append(
            dict(
                exp_name=exp_name,
                model=size,
                lambda_coord=lambda_coord,
                num_points=num_points,
                epoch=epoch,
                positive_radius=positive_radius,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                heat_gaussian=heat_gaussian,
                gaussian_sigma_ratio=gaussian_sigma_ratio,
                nms_radius=(
                    args.nms_radius[0] if len(args.nms_radius) == 1 else list(args.nms_radius)
                ),
                tooth_loss=args.tooth_loss,
                normals=normals,
                project_to_surface=project_to_surface,
                save_path=grid_root / exp_name,
                config=config_path,
                sbatch=sbatch_path,
            )
        )

    manifest_path = grid_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    submit_all = grid_root / "submit_all.sh"
    submit_all.write_text(SUBMIT_ALL_TEMPLATE)
    submit_all.chmod(0o755)

    options = [
        name
        for name, enabled in (
            ("tooth_loss", args.tooth_loss),
            ("normals", args.normals and not args.sweep_normals),
            ("sweep_normals", args.sweep_normals),
            (
                "project_to_surface",
                args.project_to_surface and not args.sweep_project_to_surface,
            ),
            ("sweep_project_to_surface", args.sweep_project_to_surface),
            (
                "heat_gaussian",
                args.heat_gaussian and not args.sweep_heat_gaussian,
            ),
            ("sweep_heat_gaussian", args.sweep_heat_gaussian),
            ("nms_radius=" + str(list(args.nms_radius)), len(args.nms_radius) > 1),
        )
        if enabled
    ]
    print(f"grid root: {grid_root}")
    print(
        f"generated {len(manifest_rows)} experiments "
        f"({len(args.models)} models x {len(args.lambda_coord)} lambda_coord "
        f"x {len(args.num_points)} num_points x {len(args.epoch)} epoch x "
        f"{len(args.positive_radius)} positive_radius x {len(args.focal_alpha)} "
        f"focal_alpha x {len(args.focal_gamma)} focal_gamma x "
        f"{len(heat_gaussian_combos)} heat_gaussian x {len(normals_values)} "
        f"normals x {len(project_values)} project_to_surface)"
        + (f", options: {', '.join(options)}" if options else "")
    )
    print(f"submit with: bash {submit_all}")


if __name__ == "__main__":
    main()
