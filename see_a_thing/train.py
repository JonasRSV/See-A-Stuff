import sys
import numpy as np
import time as time_module
import see_a_thing.files as files
import see_a_thing.graphs as graphs
import os
import sklearn.model_selection as skm
import tensorflow as tf


BATCH_SIZE = 16
EPOCHS     = 5

def fit(settings):

    if settings["overwrite"]:
        files.remove_model_directory(settings)

    datas, labels, categories = files.load_data(settings)

    data_train, data_validation, label_train, label_validation =\
            skm.train_test_split(datas, 
                                 labels,
                                 train_size=0.8,
                                 shuffle=True)


    graph, inputs, labels = graphs.GraphBuilder.of(datas.shape,
                                                   categories)

    training_feed_dicts =\
            preprocess_feed_dicts(data_train,
                                  label_train,
                                  inputs,
                                  labels,
                                  BATCH_SIZE)

    #############################################
    # Prepair session, summary writer and graph #
    #############################################

    with tf.Session() as session:
        learn_ops, summaries_ops = graph.get_learn_and_summaries_tensors()

        global_step = tf.train.get_global_step()
        summary_writer = tf.summary.FileWriter("./summaries", 
                                               session=session,
                                               graph=session.graph)

        session.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            for feed_dict in training_feed_dicts:
                _, summaries, step = session.run((learn_ops, 
                                                  summaries_ops,
                                                  global_step),
                                                 feed_dict=feed_dict)

                summary_writer.add_summary(summaries, step)


        validate_training(inputs, 
                          graph.odds, 
                          data_validation, 
                          label_validation, 
                          categories, 
                          summary_writer)

        summary_writer.flush()
        summary_writer.close()

        graphs.GraphBuilder.save_graph(settings["model_path"])

        session.close()


def validate_training(in_tensor, out_tensor, in_data, out_labels, categories, summary_writer):
    session = tf.get_default_session()

    predictions = np.argmax(session.run(out_tensor, feed_dict={in_tensor: in_data}), axis=1)

    print("Validation Score: ", sum(predictions == out_labels) / len(out_labels))
    print("Total Validation: ", len(out_labels))
    for index, category in enumerate(categories):
        print("{} Guesses {}".format(category, sum((np.ones_like(predictions) * index) == predictions)))


def preprocess_feed_dicts(a1, a2, a1t, a2t, batch):
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

