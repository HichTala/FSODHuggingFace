import argparse
import json
import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    parser.add_argument('--config', type=str, default="configs/config.json")
    parser.add_argument('--dataset_names', nargs='+', default=["detection-datasets/coco"])
    parser.add_argument('--seed', nargs='+', default=["1338"])
    parser.add_argument('--shots', nargs='+', default=["10"])
    parser.add_argument('--output_dir', nargs='+', default=["detr-finetuned-coco-10shots"])

    return parser

def build_cmd(config):
    cmd = ""
    for key, value in config.items():
        cmd += f"--{key}"
        cmd += str(value)
    return cmd

def main(args):
    with open(args.config) as f:
        config = json.load(f)

    for dataset_name in args.dataset_names:
        for seed in args.seed:
            for shots in args.shots:
                for output_dir in args.output_dir:
                    config["dataset_name"] = dataset_name
                    config["seed"] = seed
                    config["shots"] = shots
                    config["output_dir"] = output_dir

                    cmd = build_cmd(config)
                    submit_job(cmd)



def submit_job(cmd):
    subprocess.run(["sbatch", "submit.sh", cmd])

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)