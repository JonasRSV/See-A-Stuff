import os
import sys
import numpy as np

def preprocess_image(image):
    #####################
    # Preprocess images #
    #####################
    return image


def preprocess_feed_dicts(a1, a2, a1t, a2t, batch):
    #########################################
    # Preprocess two arrays into feed_dicts #
    # ARGS:                                 #
    #       A1: Array 1                     #
    #       A2: Array 2                     #
    #       A1T: Array1 Tensor              #
    #       A2T: Array2 Tensor              #
    #       BATCH: Batch Size               #
    #                                       #
    # RETURNS: List of feed_dicts           #
    #########################################
    assert len(a1) == len(a2)

    feed_dicts = []
    a_len      = len(a1)

    bi = 0
    while bi < a_len:
        a1b = a1[bi: bi + batch]
        a2b = a2[bi: bi + batch]
        feed_dict   = {a1t: a1b,
                       a2t: a2b}

        feed_dicts.append(feed_dict)

        bi += batch

    return feed_dicts


def read_training_data(path):
    ##################################################
    # Read Training Data from the Training Directory #
    # ARGS:                                          #
    #      PATH: Path to training directory          #
    #                                                #
    # RETURNS: LIST of all data                      #
    #          LIST of all labels                    #
    #          number of categories                  #
    #          Category List                         #
    ##################################################
    subject_files = os.listdir(path)

    category       = 0
    categories     = []
    subject_datas  = []
    subject_labels = []
    for subject_file in subject_files:
        subject_data = None
        subject_path = os.path.join(path, subject_file)
        try:
            with open(subject_path, "rb") as subject_file_handle:
                subject_data = np.load(subject_file_handle)
        except OSError:
            sys.stderr.write("\n" + subject_path
                    + " appears to be corrupt, ignoring it.")

        if subject_data is not None:
            images = subject_data.shape[0]

            subject_datas.append(subject_data)
            subject_labels.extend([category] * images)
            categories.append(subject_file)
            category += 1

            sys.stdout.write("\nRecovered data for {}, {} images".format(subject_file, images))

    print()
    return np.concatenate(subject_datas), subject_labels, category, categories


def check_prequisites(provided_root=None):
    #####################################
    # Check prequisites for everything  #
    #                                   #
    # ARGS:                             #
    #      PROVIDED_ROOT: Training dir  #
    #####################################

    path = None
    if provided_root:
        if not os.path.isdir(provided_root):
            sys.stderr.write(provided_root + "\nis not a valid directory"
                       + "\nplease provide an absolute path")

            sys.stderr.write("\nPrequisite Check Failed.")
            sys.stderr.write("\n\n Run with --fix flag to auto fix all prequisites")
            sys.exit(1)

        path = provided_root

    elif not os.path.isdir("./training"):
        sys.stderr.write("\nThere is not ./training directory"
                   + "\nplease provide a training path from the CLI"
                   + "\nor create the relative training directory")

        sys.stderr.write("\nPrequisite Check Failed.")
        sys.stderr.write("\n\n Run with --fix flag to auto fix all prequisites")
        sys.exit(1)
    else:
        path = "./training"

    return path


def fix_prequisites():
    #######################
    # Fix all prequisites #
    #######################
    os.mkdir("training")

    return True



def print_logo():
   print('                                                                                                                                ')
   print('\033[92m' + '         _______. _______  _______           ___           .___________. __    __   __  .__   __.   _______     __       ___      ' + '\033[0m')
   print('\033[92m' + '        /       ||   ____||   ____|         /   \          |           ||  |  |  | |  | |  \ |  |  /  _____|   /_ |     / _ \   ' + '\033[0m')
   print('\033[92m' + '       |   (----`|  |__   |  |__    ______ /  ^  \   ______`---|  |----`|  |__|  | |  | |   \|  | |  |  __      | |    | | | |  ' + '\033[0m')
   print('\033[92m' + '        \   \    |   __|  |   __|  |______/  /_\  \ |______|   |  |     |   __   | |  | |  . `  | |  | |_ |     | |    | | | |  ' + '\033[0m')
   print('\033[92m' + '    .----)   |   |  |____ |  |____       /  _____  \           |  |     |  |  |  | |  | |  |\   | |  |__| |     | |  __| |_| |  ' + '\033[0m')
   print('\033[92m' + '    |_______/    |_______||_______|     /__/     \__\          |__|     |__|  |__| |__| |__| \__|  \______|     |_| (__)\___/   ' + '\033[0m')
   print( '                                                                                                                                ')




