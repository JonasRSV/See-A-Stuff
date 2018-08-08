import sys
import numpy as np
import time as time_module
import common

def record(subject_name, camera, training_root):
    ##########################################################
    # Subject Name: Label of the recorded data               #
    # camera: A generator yielding images                    #
    ##########################################################

    images = []
    for image in camera:
        images.append(common.preprocess_image(image))

    images = np.vstack(images)

    ######################
    # Check for old data #
    ######################
    file_path = os.path.join(training_root, subject_name)

    subject_data = np.array()
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as subject_data_file:
                subject_data = np.load(subject_data_file)
        except OSError:
            sys.stderr.write(file_path 
            + " appears to be corrupt, overwriting with new data")

    images = np.concatenate([images, subject_data])

    #################
    # Write to file #
    #################
    with open(file_path, "wb") as subject_data_file:
        np.save(subject_data_file, images)

    return True


def fit(path):
    print("Totally fitting all of")
    print(os.listdir(path))

