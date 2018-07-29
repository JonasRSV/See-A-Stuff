import tensorflow as tf
import numpy as np
import model
import os
import sys
import cv2
from random import shuffle

STANDARDIZED_SIZE = (299, 299)
SEE_A_STUFF_RATE  = 10
SEE_A_STUFF_IMS   = 200

def train_model():
    if not os.path.isdir("training"):
        sys.stdout.write("Please create training file")
        sys.exit(1)

    ##############
    # Load Data. #
    ##############
    training_files = os.listdir("./training")

    train_data = []
    labels     = []
    classes    = []
    for index, cls in enumerate(training_files):
        with open("./training/{}".format(cls), "rb") as c_data:
            ims = np.load(c_data)
            lab = [index] * len(ims)

            print("{}, Samples: {}".format(cls, len(ims)))
            print(ims.shape)

            train_data.append(ims)
            labels.extend(lab)
            classes.append(cls)


    train_data = np.concatenate(train_data, axis=0)
    il         = list(zip(train_data, labels))

    ###################
    # Randomize order #
    ###################
    shuffle(il)

    t_images, t_labels = zip(*il)
    t_images = np.array(t_images)
    t_labels = np.array(t_labels)

    num_classes = len(classes)

    #############################
    # Build model and training  #
    #############################
    model_input = tf.placeholder(tf.float32, 
                                [None, 
                                 None, 
                                 None, 
                                 3])

    resize_op  = tf.image.resize_images(model_input, STANDARDIZED_SIZE)
    logits, end_points = model.xception(resize_op, 
                                        num_classes=num_classes, 
                                        is_training=True)

    BATCH_SIZE  = 10

    initial_learning_rate = 0.001
    global_step = tf.train.get_or_create_global_step()
    decay_steps = 2 * (len(t_images) // BATCH_SIZE)
    decay_rate  = 0.1

    lr = tf.train.exponential_decay(
                learning_rate = initial_learning_rate,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = decay_rate,
                staircase = True)


    labels = tf.placeholder(tf.int32, [None])
    one_hot_labels = tf.one_hot(labels, num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, 
                                           logits = logits)

    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

    ######################
    # Preprocess batches #
    ######################
    print("Preprocessing batches..")

    feed_dicts = []
    BATCHES = len(t_images) // BATCH_SIZE
    for b in range(BATCHES):
        ims = t_images[b * BATCH_SIZE: (b + 1) * (BATCH_SIZE)]
        lab = t_labels[b * BATCH_SIZE: (b + 1) * (BATCH_SIZE)]
        feed_dicts.append({model_input: ims, labels: lab})


    ################
    # Run Training #
    ################
    print("Training...")

    EPOCH   = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(EPOCH):
            EPOCH_LOSS = 0
            for b, fd in enumerate(feed_dicts):
                l, _ = sess.run((loss, optimizer), feed_dict=fd)

                print("EPOCH {}, BATCH {}, LOSS: {}".format(e, b, l))

                EPOCH_LOSS += l

            print("\nEPOCH LOSS {}\n".format(EPOCH_LOSS))
        
    saver = tf.train.Saver()
    saver.save(sess, "model/stuffseer.model")

    print("Training Done!")

def record_training_data():
    """Add training data."""
    name = input("Label: ")

    #################################
    # Load previously recorded data #
    #################################
    recorded_frames = None
    if os.path.isfile("./training/{}".format(name)):
        try:
            with open("./training/{}".format(name), "rb") as rf:
                recorded_frames = np.load(rf)
        except OSError:
            print("Could not load file ./training/{} because it was corrupt... overwriting it with recorded data".format(name))

    ###################
    # Record new data #
    ###################
    capture = cv2.VideoCapture(0)

    train_frames = []
    for i in range(SEE_A_STUFF_IMS):
        ret, frame = capture.read()
        if not i % SEE_A_STUFF_RATE:
            std_image = cv2.resize(frame, STANDARDIZED_SIZE)
            train_frames.append(std_image)

            cv2.imshow('frame', std_image)
            cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()


    ################
    # Combine data #
    ################
    if recorded_frames is None:
        train_frames = np.array(train_frames)
    else:
        train_frames = np.concatenate([np.array(train_frames), recorded_frames], axis=0)

    ##############
    # Store data #
    ##############
    with open("training/{}".format(name), "wb") as train_data_storage:
        np.save(train_data_storage, train_frames)

    print("Added training data.")

