import tensorflow as tf
import os
import shutil
import see_a_thing.modules.inception as inception


def precision_recall(logits, labels):
    score = tf.cast(tf.argmax(logits, axis=0), tf.int32) != tf.cast(tf.argmax(labels, axis=0), tf.int32)

    precision = tf.reduce_sum(tf.cast(score, tf.float32)) #/ tf.cast(tf.shape(logits)[0], tf.float32)

    # true_positives  = tf.logical_and(score, label_bools)
    # false_negatives = tf.logical_and(tf.logical_not(score), label_bools)

    # num_true_positives  = tf.reduce_sum(tf.cast(true_positives, tf.float32))
    # num_false_negatives = tf.reduce_sum(tf.cast(false_negatives, tf.float32))

    # recall = num_true_positives / (num_true_positives + num_false_negatives)

    return tf.cast(tf.shape(logits)[0], tf.float32)



def minizeption_network(x):
    x = tf.layers.conv2d(x, 64, 7, 2)
    x = tf.layers.max_pooling2d(x, 3, 2)

    x = inception.Inception(64, scope="inception_1")(x)
    x = tf.layers.max_pooling2d(x, 3, 2)
    x = inception.Inception(64, scope="inception_2")(x)
    x = tf.layers.max_pooling2d(x, 3, 2)
    x = inception.Inception(64, scope="inception_3")(x)
    x = tf.layers.max_pooling2d(x, 3, 2)

    x = tf.nn.pool(x, [8, 8], pooling_type="AVG", padding="VALID")
    features = tf.layers.flatten(x)
    return features


class GraphBuilder(object):
    INPUTS_NAME       = "inputs"
    PROBABILITES_NAME = "probabilites"
    CATEGORY_NAME     = "category"
    CATEGORIES_NAME   = "categories"


    INIT_LEARNING_RATE = 0.001
    LR_DECAY_STEPS     = 10000
    LR_DECAY_RATE      = 0.96
    LR_STAIRCASE       = False

    def of(im_dims, categories):
        _, Y, X, channels = im_dims

        labels = tf.placeholder(tf.int32, [None])
        inputs = tf.placeholder(tf.float32, 
                                    [None, Y, X, channels], 
                                    name=GraphBuilder.INPUTS_NAME)


        return GraphBuilder(inputs, labels, categories), inputs, labels


    def __init__(self, inputs, labels, categories):
        self.inputs         = inputs
        self.labels         = tf.one_hot(labels, depth=len(categories))
        self.num_categories = len(categories)
        self.categories     = tf.constant(categories, name=GraphBuilder.CATEGORIES_NAME)

    def build_classifier(self):
        features = minizeption_network(self.inputs)
        logits   = tf.layers.dense(features, self.num_categories, activation=None)
        odds     = tf.nn.softmax(logits)

        #####################
        # For the Live View #
        #####################
        category_op = tf.map_fn(lambda prob: self.categories[tf.argmax(prob)],
                                odds,
                                dtype=tf.string)

        tf.identity(category_op, name=GraphBuilder.CATEGORY_NAME)
        tf.identity(odds, name=GraphBuilder.PROBABILITES_NAME)

        return logits, odds

    def get_learn_and_summaries_tensors(self):

        logits, self.odds = self.build_classifier()
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.labels,
                    logits=logits))

        global_step = tf.train.create_global_step()
        lr          = tf.train.exponential_decay(GraphBuilder.INIT_LEARNING_RATE,
                                                 global_step, 
                                                 GraphBuilder.LR_DECAY_STEPS,
                                                 GraphBuilder.LR_DECAY_RATE,
                                                 staircase=GraphBuilder.LR_STAIRCASE)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)\
                .minimize(loss, 
                          global_step=global_step)


        loss_summary = tf.summary.scalar("training/loss", loss)
        lr_summary   = tf.summary.scalar("training/learning rate", lr)

        training_summaries = tf.summary.merge([loss_summary, lr_summary])

        VALIDATE_VISUALIZE = 3

        val_batch  = tf.shape(self.inputs)[0]
        validate_indexes = tf.random_uniform([VALIDATE_VISUALIZE], minval=0, maxval=val_batch, dtype=tf.int32) 
        val_summary_ims  = tf.summary.image("validate", tf.gather(self.inputs, validate_indexes), max_outputs=VALIDATE_VISUALIZE)

        precision = precision_recall(logits, self.labels)

        val_summary_precision = tf.summary.scalar("validate/precision", precision)
        # val_summary_recall    = tf.summary.scalar("validate/recall", recall)

        validation_summaries = tf.summary.merge([val_summary_ims, val_summary_precision])

        return optimizer, training_summaries, validation_summaries 

    def save_graph(model_directory):
        builder = tf.saved_model.builder.SavedModelBuilder(model_directory)
        builder.add_meta_graph_and_variables(tf.get_default_session(), ["see-a-thing-tag"])
        builder.save()

    def restore_graph(model_directory):
        meta_graph = tf.saved_model.loader.load(tf.get_default_session(),
                                                ["see-a-thing-tag"],
                                                model_directory)

        graph = tf.get_default_graph()

        category_tensor     = graph.get_tensor_by_name(GraphBuilder.CATEGORY_NAME + ":0")
        probabilites_tensor = graph.get_tensor_by_name(GraphBuilder.PROBABILITES_NAME + ":0")
        categories          = graph.get_tensor_by_name(GraphBuilder.CATEGORIES_NAME + ":0")
        input_tensor        = graph.get_tensor_by_name(GraphBuilder.INPUTS_NAME + ":0")
        
        return input_tensor, category_tensor, probabilites_tensor, categories
