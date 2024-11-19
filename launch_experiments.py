import argparse
import json
import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    return parser

def main(args):
    with open("config.json") as f:
        config = json.load(f)

    cmd = ""
    for key, value in config.items():
        cmd += f"--{key}"
        cmd += str(value)

def submit_job(cmd):
    subprocess.run(["sbatch", "submit.sh", cmd])

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)