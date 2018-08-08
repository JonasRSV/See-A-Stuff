import argparse
import see_a_thing as sat
import sys


DEFAULT_TIME = 60
DEFAULT_FREQ = 1

########################
# Don't touch this one #
########################
DEFAULT_PATH = "training"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", 
                        help="record labeled data with: --record XXX",
                        type=str)

    parser.add_argument("-f", "--frequency",
                        help="Frequency camera should run at in Hz",
                        type=float)

    parser.add_argument("-t", "--time",
                       help="Time the camera should run for (In seconds)",
                       type=int)

    parser.add_argument("-d", "--display",
                        help="Display what is being recorded",
                        action="store_true")

    parser.add_argument("-l", "--live",
                        help="Run the detector live",
                        action="store_true")

    parser.add_argument("--train",
                        help="Train the model",
                        action="store_true")

    parser.add_argument("--fix",
                        help="\'Fix all prequisites for me\'",
                        action="store_true")

    parser.add_argument("-p", "--path", 
                        help="Path to the training directory",
                        type=str)

    args = parser.parse_args()

    if args.fix:
        sat.common.fix_prequisites()

    sat.common.check_prequisites(args.path)

    path = args.path if args.path else DEFAULT_PATH

    if args.train:
        train(path)

    if args.live:
        live(args.time, args.frequency, args.display)

    frequency = args.frequency if args.frequency else DEFAULT_FREQ
    time      = args.time if args.time else DEFAULT_TIME

    if args.record:
        record(args.record, time, frequency, args.display, path)


def record(label, time, frequency, display, path):
    #####################################
    # Record the data used for training #
    #####################################
    camera_feed = sat.camera.camera_feed(time, frequency, display)
    sat.train.record(label, camera_feed, path)
    sys.exit(0)

def live(time, frequency, display):
    print("Live at freq {} for {}".format(frequency, time))

    ###########################################
    # Update frequency of 10 times per second #
    ###########################################
    if not frequency:
        frequency = 10

    ############
    # One Hour #
    ############
    if not time:
        time = 3600

    camera_feed    = sat.camera.camera_feed(time, frequency, display)
    predictor_feed = sat.live.predictor_feed(camera_feed)

    sat.live.demo_display(predictor_feed)
    sys.exit(0)

def train(path):
    sat.train.fit(path)
    sys.exit(0)


if __name__ == "__main__":
    main()
