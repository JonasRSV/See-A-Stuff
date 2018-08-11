import tensorflow as tf
import os
import shutil

GRAPH_NAME         = None
GRAPH_STORAGE      = None

GRAPH_INPUT        = None
GRAPH_LABELS       = None
GRAPH_OUTPUT       = None
CATEGORIES         = None
GLOBAL_STEP_TENSOR = None

INIT_LEARNING_RATE = 0.1
LR_DECAY_STEPS     = 10000
LR_DECAY_RATE      = 0.96
LR_STAIRCASE       = True

MODEL_DIRECTORY    = "./model"

def create_graph_placeholders(data_shape, categories):
    global GRAPH_INPUT, GRAPH_LABELS, CATEGORIES
    _, Y, X, channels = data_shape

    CATEGORIES = categories

    label_placeholder = tf.placeholder(tf.int32, [None])
    GRAPH_INPUT       = tf.placeholder(tf.float32, [None, Y, X, channels], name="input")
    GRAPH_LABELS      = tf.one_hot(label_placeholder, depth=categories)

    return GRAPH_INPUT, label_placeholder


def build_classifier():

    flat = tf.layers.flatten(GRAPH_INPUT)
    out = tf.layers.dense(flat, CATEGORIES, activation=tf.nn.softmax)

    return out


def get_learn_and_summaries_tensors():
    global GRAPH_OUTPUT, GRAPH_LABELS

    GRAPH_OUTPUT = build_classifier()

    loss = tf.losses.softmax_cross_entropy(GRAPH_LABELS,
                                           GRAPH_OUTPUT)

    global_step = tf.train.create_global_step()
    lr          = tf.train.exponential_decay(INIT_LEARNING_RATE,
                                             global_step, 
                                             LR_DECAY_STEPS,
                                             LR_DECAY_RATE,
                                             staircase=LR_STAIRCASE)


    optimizer = tf.train.AdamOptimizer(learning_rate=lr)\
            .minimize(loss, 
                      global_step=global_step)


    tf.summary.scalar("loss", tf.reduce_mean(loss))
    tf.summary.scalar("learning rate", lr)

    return optimizer, tf.summary.merge_all()


def save_graph(categories):
    categories   = tf.constant(categories, name="categories")
    probabilites = tf.identity(GRAPH_OUTPUT, name="probabilites")

    category     = tf.identity(tf.map_fn(lambda probs: categories[tf.argmax(probs)], 
                                         probabilites, dtype=tf.string),
                               name="category")

    if os.path.isdir(MODEL_DIRECTORY):
        shutil.rmtree(MODEL_DIRECTORY)


    builder = tf.saved_model.builder.SavedModelBuilder(MODEL_DIRECTORY)
    builder.add_meta_graph_and_variables(tf.get_default_session(), ["see-a-thing-tag"])
    builder.save()

def restore_graph():
    meta_graph = tf.saved_model.loader.load(tf.get_default_session(),
                                            ["see-a-thing-tag"],
                                            MODEL_DIRECTORY)

    graph = tf.get_default_graph()

    category_tensor     = graph.get_tensor_by_name("category:0")
    probabilites_tensor = graph.get_tensor_by_name("probabilites:0")
    categories          = graph.get_tensor_by_name("categories:0")
    input_tensor        = graph.get_tensor_by_name("input:0")
    

    return input_tensor, category_tensor, probabilites_tensor, categories
