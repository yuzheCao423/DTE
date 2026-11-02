import argparse
import json

def load_config():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    argparser.add_argument("-id", "--expid", type=int, required=True)
    args = argparser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    return config, args.expid