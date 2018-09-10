import sys
import os
import numpy as np
import time as time_module
import see_a_thing.utils.files as files
import see_a_thing.graphs as graphs
import tensorflow as tf


train_settings = {"batch_size": 16,
                  "epochs": 100}

def train(settings):

    if settings["overwrite"]:
        files.remove_model_directory(settings)

    datas, labels, categories = files.load_data(settings)
    graph, inputs_t, labels_t = graphs.GraphBuilder.of(datas.shape,
                                                       categories)


    val_feed, train_feed_gen = get_val_and_train_feed(datas, labels, inputs_t, labels_t)

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

        for ep in range(train_settings["epochs"]):
            train_dicts = next(train_feed_gen)

            summary = tf.Summary()

            for train_dict in train_dicts:
                _, summaries, step = session.run((learn_ops, 
                                                  summaries_ops,
                                                  global_step),
                                                 feed_dict=train_dict)

                summary_writer.add_summary(summaries, step)


        summary_writer.flush()
        summary_writer.close()

        graphs.GraphBuilder.save_graph(settings["model_path"])

        session.close()


def validate(val_feed, epoch):
    print("Todo: Validation")


def get_val_and_train_feed(datas, labels, inputs_t, labels_t):
    dsize = len(datas)

    indexes = np.arange(dsize)

    np.random.shuffle(indexes)

    train_indexes    = indexes[int(dsize * 0.1):]
    validate_indexes = indexes[:int(dsize * 0.1)]

    val_dict = {inputs_t: datas[validate_indexes], 
                labels_t: labels[validate_indexes]}

    def train_feed_gen():
        while True:
            feed_dicts = []

            np.random.shuffle(train_indexes)

            for d, l in zip(np.array_split(datas[train_indexes], dsize // train_settings["batch_size"]),
                            np.array_split(labels[train_indexes], dsize // train_settings["batch_size"])):
                train_dict = {inputs_t: d,
                              labels_t: l}

                feed_dicts.append(train_dict)

            yield feed_dicts

    return val_dict, train_feed_gen()
