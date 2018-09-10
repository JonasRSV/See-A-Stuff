import numpy as np
import os
import shutil
import sys

def append_data(settings, images):
    file_path = os.path.join(settings["training_path"], settings["label"])

    previous_data = None
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as subject_data_file:
                previous_data = np.load(subject_data_file)
        except OSError:
            sys.stderr.write(file_path 
            + " Ignoring and overwriting with new data.")


    if previous_data is not None:
        images = np.concatenate([images, previous_data])

    #################
    # Write to file #
    #################
    with open(file_path, "wb") as subject_data_file:
        np.save(subject_data_file, images)


def load_data(settings):
    data_files = os.listdir(settings["training_path"])

    categories     = []
    datas          = []
    labels         = []
    for data_file in data_files:
        data = None
        path = os.path.join(settings["training_path"], data_file)
        try:
            with open(path, "rb") as file_handle:
                data = np.load(file_handle)
        except OSError:
            sys.stderr.write("\n" + path
                    + " appears to be corrupt, ignoring it.")

        if data is not None:
            num_images = data.shape[0]

            datas.append(data)
            labels.extend([len(categories)] * num_images)
            categories.append(data_file)

            sys.stdout.write("\nRecovered data for {}, {} images"
                    .format(data_file, num_images))

    print()
    return np.concatenate(datas), np.array(labels), categories

def check_file_prequisites(settings):


    file_check_scenairos = np.array([check_file_prequisites_record,
                                     check_file_prequisites_train,
                                     check_file_prequisites_live,
                                     check_file_prequisites_live])

    for check in file_check_scenairos[settings["options"]]:
        check(settings)


def check_file_prequisites_train(settings):
    check_training_directory(settings)

    model_root    = settings["model_path"]
    if os.path.isdir(model_root):
        if len(os.listdir(model_root)) and not settings["overwrite"]:
            sys.stderr.write("\n" + model_root + "\nis not empty and --overwrite flag"
                            + " was not provided, please provide an empty directory or"
                            + " overwrite using the --overwrite flag.")

            sys.exit(1)


# TODO: Add more Comprehensive check
def check_file_prequisites_live(settings):

    model_root = settings["model_path"]

    if not os.path.isdir(model_root):
        sys.stderr.write("\n" + model_root + "\n is not a valid directory")
        sys.exit(1)


def check_file_prequisites_record(settings):
    check_training_directory(settings)


def check_training_directory(settings):
    training_root = settings["training_path"]

    if not os.path.isdir(training_root):
        sys.stderr.write("\n" + provided_root + "\nis not a valid directory"
                   + "\nplease provide an absolute path")

        sys.stderr.write("\nPrequisite Check Failed.")
        sys.stderr.write("\n\n Run with --fix flag to auto fix all prequisites")
        sys.exit(1)


def fix_file_prequisites(settings):
    recreate_training_directory(settings)


def remove_model_directory(settings):
    if os.path.isdir(settings["model_path"]):
        shutil.rmtree(settings["model_path"])


def recreate_training_directory(settings):
    if os.path.isdir(settings["training_path"]):
        shutil.rmtree(settings["training_path"])

    os.mkdir(settings["training_path"])


def clean_files(settings):
    remove_model_directory(settings)
    recreate_training_directory(settings)
    print("\nFiles Cleaned! :)")


# TODO: Make it more specialized?
def display_data(settings):
    load_data(settings)
