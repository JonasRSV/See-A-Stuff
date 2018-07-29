import tensorflow as tf
import numpy as np
import model
import os
import sys
import cv2
from random import shuffle

STANDARDIZED_SIZE = (299, 299)
SEE_A_STUFF_RATE  = 10
SEE_A_STUFF_IMS   = 2000

def train_model():
    if not os.path.isdir("training"):
        sys.stdout.write("Please create training file")
        sys.exit(1)

    training_files = os.listdir("./training")

    train_data = []
    labels     = []
    classes    = []
    for index, cls in enumerate(training_files):
        with open("./training/{}".format(cls), "rb") as c_data:
            ims = np.load(c_data)
            lab = [index] * len(ims)

            print("{}, Samples: {}".format(cls, len(ims)))

            train_data.append(ims)
            labels.extend(lab)
            classes.append(cls)


    train_data = np.concatenate(train_data, axis=0)
    il         = list(zip(train_data, labels))

    shuffle(il)

    t_images, t_labels = zip(*il)
    t_images = np.array(t_images)
    t_labels = np.array(t_labels)

    num_classes = len(classes)

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


    labels = tf.placeholder(tf.int32, [None, 1])
    one_hot_labels = tf.one_hot(labels, num_classes)
    one_hot_labels = tf.squeeze(one_hot_labels, [1])
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, 
                                           logits = logits)

    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        EPOCH   = 1000
        BATCHES = len(t_images) // BATCH_SIZE
        for e in range(EPOCH):
            for b in range(BATCHES):
                ims = t_images[b * BATCH_SIZE:b * (BATCH_SIZE + 1)]
                lab = np.reshape(t_labels[b * BATCH_SIZE: b * (BATCH_SIZE + 1)], (-1, 1))

                feed_dict={model_input: ims, labels: lab}
                l, _ = sess.run((loss, optimizer), feed_dict=feed_dict)

                print("EPOCH {}, BATCH {}, LOSS: {}".format(e, b, l))
        
    saver = tf.train.Saver()
    saver.save(sess, "model/stuffseer.model")

    print("Training Done!")


def see_a_stuff():
    name = input("stuff: ")

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


    training_frames = np.array(train_frames)
    with open("training/{}".format(name), "wb") as see_a_thing_data:
        np.save(see_a_thing_data, training_frames)

    print("saw a stuff")


def see_people():

    capture = cv2.VideoCapture(0)
    try:
        while True:
            # TODO: ADD MODEL
            ret, frame = capture.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


def see_sobel():
    capture = cv2.VideoCapture(0)
    try:
        while True:
            # TODO: ADD MODEL
            ret, frame = capture.read()

            laplacian = cv2.Laplacian(frame,cv2.CV_64F)
            cv2.imshow('frame', laplacian)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if "-train" in sys.argv:
        train_model()

    if "-sas" in sys.argv:
        see_a_stuff()

    if "-see" in sys.argv:
        see_people()

    if "-ss" in sys.argv:
        see_sobel()












