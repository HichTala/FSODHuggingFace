import argparse
import json
import subprocess


def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    parser.add_argument('--config', type=str, default="configs/config.json")
    parser.add_argument('--dataset_names', nargs='+', default=["detection-datasets/coco"])
    parser.add_argument('--seed', nargs='+', default=["1338"])
    parser.add_argument('--shots', nargs='+', default=["10"])
    parser.add_argument('--output_dir', type=str, default="detr-finetuned")

    parser.add_argument('--freeze_modules', type=str)
    parser.add_argument('--freeze_at', type=str)

    parser.add_argument('--exec_type', type=str, default="slurm")

    return parser


def build_cmd(config):
    cmd = ""
    for key, value in config.items():
        cmd += f" --{key} "
        cmd += str(value)
    return cmd


def submit_job(cmd, exec_type):
    if exec_type == "python":
        cmd = f"python run_object_detection.py{cmd}"
    subprocess.run(cmd, shell=True)


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    if not args.freeze_modules:
        args.freeze_modules = [[], ['backbone'], ['backbone'], ['bias'], ['norm']]
    if args.freeze_at:
        args.freeze_at = [[], ['0'], ['half'], [], []]

    for dataset_name in args.dataset_names:
        for seed in args.seed:
            for shot in args.shots:
                for freeze_modules, freeze_at in zip(args.freeze_modules, args.freeze_at):
                    output_dir = f"{args.output_dir}/{dataset_name.split('/')[-1]}/{seed}/{shot}/"

                    if len(freeze_modules) == 0:
                        output_dir += "full_finetuning"
                    for freeze_module, freeze_at_single in zip(freeze_modules, freeze_at):
                        if output_dir[-1] != '/':
                            output_dir += '_'
                        output_dir += f"{freeze_module}-{freeze_at_single}" if freeze_at_single != '0' else f"{freeze_module}-full"

                    config['freeze_modules'] = freeze_modules
                    config['freeze_at'] = freeze_at

                    config["dataset_name"] = dataset_name
                    config["seed"] = seed
                    config["shots"] = shot
                    config["output_dir"] = output_dir

                    cmd = build_cmd(config)
                    submit_job(cmd, exec_type=args.exec_type)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
