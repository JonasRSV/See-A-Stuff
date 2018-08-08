import os
import sys

def preprocess_image(image):
    return image

def check_prequisites(provided_root=None):

    if provided_root:
        if not os.path.isdir(provided_root):
            sys.stderr.write(provided_root + "\nis not a valid directory"
                       + "\nplease provide an absolute path")

            sys.stderr.write("\nPrequisite Check Failed.")
            sys.stderr.write("\n\n Run with --fix flag to auto fix all prequisites")
            sys.exit(1)

    if not os.path.isdir("./training"):
        sys.stderr.write("\nThere is not ./training directory"
                   + "\nplease provide a training path from the CLI"
                   + "\nor create the relative training directory")

        sys.stderr.write("\nPrequisite Check Failed.")
        sys.stderr.write("\n\n Run with --fix flag to auto fix all prequisites")
        sys.exit(1)

    return True

def fix_prequisites():
    os.mkdir("training")

    return True







