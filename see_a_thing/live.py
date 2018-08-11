import see_a_thing.graphs as graphs
import see_a_thing.common as common
import tensorflow as tf
import sys


def predictor_feed(camera_feed):
    with tf.Session() as session:
        input_tensor, category_tensor, probabilites_tensor, categories =\
                graphs.restore_graph()

        categories = list(map(lambda x: x.decode("utf-8"), session.run(categories)))
        try:
            while True:
                unprocessed_image = next(camera_feed)
                image = common.preprocess_image(unprocessed_image)

                prob, category = session.run((probabilites_tensor,
                                              category_tensor),
                                             feed_dict={input_tensor: [image]})
                                          

                prob     = prob[0]
                category = category[0].decode("utf-8")
                yield category, zip(categories, prob), unprocessed_image

        except StopIteration:
            sys.stderr.write("\n Camera feed done, qutting predictor feed.")

def demo_display(predictor_feed):

    try:
        while True:
            category, probabilites, _ = next(predictor_feed)

            print(category)
            print(list(probabilites))
            print()
    except StopIteration:
        sys.stderr.write("\n Predictor feed done, qutting Demo feed.")

    print("Totally Demoing")
