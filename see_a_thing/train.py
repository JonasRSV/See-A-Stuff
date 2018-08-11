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

    subject_datas, subject_labels, num_categories, categories =\
            common.read_training_data(path)

    data_train, data_validation, label_train, label_validation =\
            skm.train_test_split(subject_datas, 
                                 subject_labels,
                                 train_size=0.8,
                                 shuffle=True)


    inputs, labels =\
            graphs.create_graph_placeholders(subject_datas.shape,
                                             num_categories)

    validation_feed_dict = {inputs: data_validation,
                            labels: label_validation}

    training_feed_dicts =\
            common.preprocess_feed_dicts(data_train,
                                         label_train,
                                         inputs,
                                         labels,
                                         BATCH_SIZE)

    #############################################
    # Prepair session, summary writer and graph #
    #############################################

    with tf.Session() as session:
        graph   = tf.get_default_graph()
        learn_ops, summaries_ops = graphs.get_learn_and_summaries_tensors()

        global_step = tf.train.get_global_step()

        summary_writer = tf.summary.FileWriter("./summaries", 
                                               session=session)

        session.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            epoch_summary = tf.Summary()
            epoch_summary.value.add(tag="epoch", simple_value=float(epoch))

            for feed_dict in training_feed_dicts:
                _, summaries, step = session.run((learn_ops, 
                                                  summaries_ops,
                                                  global_step),
                                                 feed_dict=feed_dict)

                summary_writer.add_summary(summaries, step)

            summary_writer.add_summary(epoch_summary, epoch)

        summary_writer.flush()
        summary_writer.close()
        graphs.save_graph(categories)
        session.close()


def validate_training(in_tensor, out_tensor, in_data, out_labels, summary_writer):
    pass

