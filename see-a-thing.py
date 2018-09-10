import numpy as np
import argparse
from see_a_thing.utils import common
from see_a_thing.utils import files
from see_a_thing import camera
from see_a_thing import gym
from see_a_thing import stage
import sys


DEFAULT_TIME    = 86000
DEFAULT_FREQ    = 1

DEFAULT_TRAIN_PATH  = "training"
DEFAULT_MODEL_PATH  = "model"

DEFAULT_LABEL = "smith"

FUNCTIONS = np.array([camera.record, gym.train, stage.monitor, stage.serve])


def main():
    common.print_logo()

    args    = get_cli_args()
    options = [args.record, args.train, args.commandline, args.websocket] 

    settings = {"frequency": args.max_frequency if args.max_frequency else DEFAULT_FREQ,
                "time": args.time if args.time else DEFAULT_TIME,
                "display": args.display,
                "label": args.label if args.label else DEFAULT_LABEL,
                "training_path": args.training_path if args.training_path else DEFAULT_TRAIN_PATH,
                "model_path": args.model_path if args.model_path else DEFAULT_MODEL_PATH,
                "options": options,
                "overwrite": args.overwrite
                }

    if args.fix:
        common.fix_prequisites(settings)

    if args.clean:
        files.clean_files(settings)

    common.check_prequisites(settings)

    for function in FUNCTIONS[options]:
        function(settings)

    if args.data:
        files.load_data(settings)


def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", "-r", choices=["record", "train"],
                        help="The recorder records the data used for training\nThe train creates the model that get served.")

    parser.add_argument("--serve", "-s", choices=["commandline", "websocket"])

    parser.add_argument("--clean",
                        help="Clean the model and training directory",
                        action="store_true")

    parser.add_argument("--label",
                        help="Label for the recorded data",
                        type=str)

    parser.add_argument("-f", "--max_frequency",
                        help="Max Frequency camera will run at in Hz",
                        type=float)

    parser.add_argument("-t", "--time",
                       help="Time the camera should run for (In seconds)",
                       type=int)

    parser.add_argument("-d", "--display",
                        help="Display what is being recorded",
                        action="store_true")

    parser.add_argument("--overwrite",
                        help="Will overwrite existing model when combined with train",
                        action="store_true")

    parser.add_argument("--fix",
                        help="\'Fix all prequisites for me\' Use with caution, will remove training data.",
                        action="store_true")

    parser.add_argument("--training_path", 
                        help="(Optional) Path to the training directory",
                        type=str)

    parser.add_argument("--model_path",
                        help="(Optional) Path to the model directory",
                        type=str)

    parser.add_argument("--data",
                        help="Print Available Data",
                        action="store_true")

    args = parser.parse_args()

    args.commandline = True if args.serve == "commandline" else False
    args.websocket   = True if args.serve == "websocket" else False
    args.record      = True if args.run   == "record" else False
    args.train       = True if args.run   == "train" else False

    return args


if __name__ == "__main__":
    main()
