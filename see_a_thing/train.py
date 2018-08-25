import sys
import os
import numpy as np
import time as time_module
import see_a_thing.utils.files as files
import see_a_thing.graphs as graphs
import tensorflow as tf
import pandas as pd


train_settings = {"batch_size": 16,
                  "epochs": 5}

def fit(settings):

    if settings["overwrite"]:
        files.remove_model_directory(settings)

    datas, labels, categories = files.load_data(settings)
    graph, inputs_t, labels_t = graphs.GraphBuilder.of(datas.shape,
                                                       categories)


    val_feed, train_gen_feed = get_val_and_train_feed(datas, labels, inputs_t, labels_t)

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
            for train_dict in train_dicts:
                _, summaries, step = session.run((learn_ops, 
                                                  summaries_ops,
                                                  global_step),
                                                 feed_dict=train_dict)

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


def validate(inputs, val_dict, graph, summary_writer)
    print("Todo: Validation")


def get_val_and_train_feed(datas, labels, inputs_t, labels_t):
    df = pd.DataFrame(data={"datas": datas, "labels": labels})

    val_set = df.sample(frac=0.1)
    df.drop(val_set.index)

    val_dict = {inputs_t: val_set["datas"], 
                labels_t: val_set["labels"]}

    def train_feed_gen():
        while True:
            feed_dicts = []

            groups   = np.arange(len(df)) // train_settings["batch_size"]
            shuffled = df.sample(frac=1)

            for _, g in shuffled.groupby(groups):
                train_dict = {inputs_t: g["datas"],
                              labels_t: g["labels"]}

                feed_dicts.append(train_dict)

            yield feed_dicts

    return val_dict, train_feed_gen
