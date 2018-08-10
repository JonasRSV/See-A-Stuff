import tensorflow as tf

GRAPH_NAME         = None
GRAPH_STORAGE      = None

GRAPH_INPUT        = None
GRAPH_LABELS       = None
GRAPH_OUTPUT       = None
OUTPUTS_SHAPE      = None
GLOBAL_STEP_TENSOR = None

INIT_LEARNING_RATE = 0.1
LR_DECAY_STEPS     = 10000
LR_DECAY_RATE      = 0.96
LR_STAIRCASE       = True

def create_graph_placeholders(data_shape, label_shape):
    global GRAPH_INPUT, GRAPH_LABELS, OUTPUTS_SHAPE
    _ Y, X, channels = data_shape
    _, categories    = label_shape

    GRAPH_INPUT   = tf.placeholder(tf.float32, [None, Y, X, channels])
    GRAPH_LABELS  = tf.placeholder(tf.float32, [None, categories])
    OUTPUTS_SHAPE = [None, categories]

    return GRAPH_INPUT, GRAPH_LABELS


def build_classifier():
    return tf.get_variable("IN THE MEAN TIME", 
                           shape=OUTPUTS_SHAPE,
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer()), None


def get_learn_and_summaries_tensors():
    global GRAPH_OUTPUT, GRAPH_LABELS

    GRAPH_OUTPUT, summaries = build_classifier()

    loss = tf.train.losses.categorical_crossentropy_v2(GRAPH_LABELS, 
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


    loss_summary = tf.summary.scalar("loss", tf.reduce_mean(loss))
    lr_summary   = tf.summary.scalar("learning rate", lr)

    summaries.append(loss_summary)
    summaries.append(lr_summary)

    return optimizer, summaries


def save_graph():
    pass


def restore_graph():
    pass
