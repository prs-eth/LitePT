from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.train import TRAINERS
from engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()

    print("options: ", args.options)
    cfg = default_config_parser(args.config_file, args.options)

    print('lauch =========================')
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
