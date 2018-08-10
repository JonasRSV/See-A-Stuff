import sys
import numpy as np
import time as time_module
import see_a_thing.common as common
import see_a_thing.graphs as graphs
import os
import sklearn.model_selection as skm
import tensorflow as tf


def record(subject_name, camera, training_root):
    ##########################################################
    # Subject Name: Label of the recorded data               #
    # camera: A generator yielding images                    #
    ##########################################################

    images = []
    for image in camera:
        images.append(common.preprocess_image(image))

    images = np.array(images)
    ######################
    # Check for old data #
    ######################
    file_path = os.path.join(training_root, subject_name)

    subject_data = None
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as subject_data_file:
                subject_data = np.load(subject_data_file)
        except OSError:
            sys.stderr.write(file_path 
            + " appears to be corrupt, overwriting with new data")

    if subject_data is not None:
        images = np.concatenate([images, subject_data])

    #################
    # Write to file #
    #################
    with open(file_path, "wb") as subject_data_file:
        np.save(subject_data_file, images)

    return True

BATCH_SIZE = 32
EPOCHS     = 10

def fit(path):
    subject_files = os.listdir(path)

    subject_labels = []
    subject_datas  = []
    for subject_file in subject_files:
        subject_data = None
        subject_path = os.path.join(path, subject_file)
        try:
            with open(subject_data, "rb") as subject_file_handle:
                subject_data = np.load(subject_file_handle)
        except OSError:
            sys.stderr.write("\n" + subject_path
                    + " appears to be corrupt, ignoring it.")

        if subject_data:
            images = subject_data.shape[0]

            subject_datas.append(subject_data)
            subject_labels.extend([subject_file] * images)

            sys.stdout.write("\nRecovered data for {}, {} images".format(subject_file, images))

    data_train, data_validation, label_train, label_validation =\
            skm.train_test_split(subject_data, 
                                 subject_labels,
                                 train_size=0.8,
                                 shuffle=True)

    ######################
    # Preprocess batches #
    ######################
    valdation_data_batch   = np.concatenate(data_validation)
    validation_label_batch = np.concatenate(label_validation)

    inputs, labels =\
            graphs.create_graph_placeholders(valdation_data_batch.shape,
                                             validation_label_batch.shape)

    validation_feed_dict = {inputs: valdation_data_batch,
                            labels: validation_label_batch}

    training_feed_dicts = []

    bi = 0
    while bi < len(data_train):
        data_batch  = data_train[bi: bi + BATCH_SIZE]
        label_batch = label_train[bi: bi + BATCH_SIZE]
        feed_dict   = {inputs: data_batch,
                       labels: label_batch}

        training_feed_dicts.append(feed_dict)

        bi += BATCH_SIZE

    #############################################
    # Prepair session, summary writer and graph #
    #############################################

    session = tf.Session()
    graph   = tf.get_default_graph()
    learn_ops, summaries_ops = graphs.get_learn_and_summaries_tensors()

    summary_writer = tf.summary.FileWriter("./summaries", 
                                           graph=graph, 
                                           session=session)

    for epoch in range(EPOCHS):
        epoch_summary = tf.Summary()
        epoch_summary.value.add(tag="epoch", simple_value=float(epoch))

        for feed_dict in training_feed_dicts:
            _, summaries, step = session.run((learn_ops, 
                                              summaries_ops,
                                              tf.train.get_gloget_global_step(graph)),
                                             feed_dict=feed_dict)

            for summary in summaries:
                summary_writer.add_summary(summary, step)

        summary_writer.add_summary(epoch_summary, epoch)

    graphs.save_graph()
    session.close()

