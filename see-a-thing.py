import numpy as np
import argparse
import see_a_thing as sat
import sys


DEFAULT_TIME    = 20
DEFAULT_FREQ    = 1

DEFAULT_TRAIN_PATH  = "training"
DEFAULT_MODEL_PATH  = "model"

DEFAULT_LABEL = "smith"

FUNCTIONS = np.array([sat.camera.record, sat.train.fit, sat.live.monitor, sat.live.serve])


def main():
    sat.common.print_logo()

    args    = get_cli_args()
    options = [args.record, args.train, args.commandline_gui, args.serve] 

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
        sat.common.fix_prequisites(settings)

    if args.clean:
        sat.files.clean_files()

    sat.common.check_prequisites(settings)

    for function in FUNCTIONS[options]:
        function(settings)


    if args.data:
        sat.files.load_data(settings)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", 
                        help="Run the recorder (Use to --label flag to specify label of the data)",
                        action="store_true")

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

    parser.add_argument("-cg", "--commandline_gui",
                        help="Run monitor with commandline gui",
                        action="store_true")

    parser.add_argument("-s", "--serve",
                        help="Serve predictions through websocket",
                        action="store_true")

    parser.add_argument("--train",
                        help="Train the model",
                        action="store_true")

    parser.add_argument("--overwrite",
                        help="Will overwrite existing model when combined with train",
                        action="store_true")

    parser.add_argument("--fix",
                        help="\'Fix all prequisites for me\'",
                        action="store_true")

    parser.add_argument("-tp", "--training_path", 
                        help="(Optional) Path to the training directory",
                        type=str)

    parser.add_argument("-mp", "--model_path",
                        help="(Optional) Path to the model directory",
                        type=str)

    parser.add_argument("--data",
                        help="Print Available Data",
                        action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
