import see_a_thing.utils.common as common
import see_a_thing.graphs as graphs
import see_a_thing.camera as camera
import json
import asyncio
import websockets
import tensorflow as tf
import curses
import sys


def get_pred_feed(cam_feed, pred_graph):
    #######################
    # Feed of predictions #
    #######################
    session = tf.get_default_session()

    inputs, category, probs, categories = pred_graph
    categories = common.decode_bytes(session.run(categories))

    def pred_feed():
        try:
            while True:
                cam_image = next(cam_feed)

                img  = common.preprocess_image(cam_image)
                p, c = session.run((probs,
                                    category),
                                    feed_dict={inputs: [img]})
                      

                p = p[0]
                c = common.decode_bytes(c)[0]

                yield c, p
        except StopIteration:
            pass

    return pred_feed(), categories


def monitor(settings):
    ##################
    # Minimal CLI UI #
    ##################
    cam_feed = camera.camera_feed(settings)
    stdscr   = curses.initscr()

    curses.noecho()
    curses.cbreak()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as session:
        # My GPU does not like integers and strings :( so detection has to run on CPU 
        with tf.device("cpu:0"):
            pred_graph = graphs.GraphBuilder.restore_graph(settings["model_path"])
            pred_feed, categories = get_pred_feed(cam_feed, pred_graph)

            try:
                while True:
                    pred, probabilites = next(pred_feed)

                    stdscr.addstr(0, 0, "{} is in front of me".format(pred))
                    for i, (cat, prob) in enumerate(zip(categories, probabilites)):
                        stdscr.addstr(i + 1, 0, "{} {}%".format(cat, round(prob * 100, 2)))
                    stdscr.refresh()
            except (StopIteration, KeyboardInterrupt):
                pass

            curses.echo()
            curses.nocbreak()
            curses.endwin()


def serve(settings):
    ########################
    # Serve from websocket #
    ########################
    cam_feed = camera.camera_feed(settings)
    stdscr   = curses.initscr()

    curses.noecho()
    curses.cbreak()

    try:
        with tf.Session() as session:
            # My GPU does not like integers and strings :( so detection has to run on CPU 
            with tf.device("cpu:0"):
                pred_graph = graphs.GraphBuilder.restore_graph(settings["model_path"])
                pred_feed, categories = get_pred_feed(cam_feed, pred_graph)

                message = {"category": None,
                           "categories": categories,
                           "probabilites": []}

                @asyncio.coroutine
                def pred_server(websocket, path):
                    while True:
                        stdscr.addstr(0, 0, "Serving clients at 0.0.0.0:5000")
                        stdscr.addstr(1, 0, "Served Client at {}".format(websocket.origin))
                        stdscr.addstr(2, 0, "{} is infront of me".format(message["category"]))
                        stdscr.refresh()
                        yield from websocket.send(json.dumps(message))
                        yield from asyncio.sleep(1)

                @asyncio.coroutine
                def feed_update():
                    while True:
                        pred, probabilites = next(pred_feed)
                        message["category"]     = common.unnumpyfy(pred)
                        message["probabilites"] = common.unnumpyfy(probabilites)

                        yield from asyncio.sleep(0.01)

                server = websockets.serve(pred_server, "0.0.0.0", 5000)

                asyncio.get_event_loop().run_until_complete(server)
                asyncio.async(feed_update())
                asyncio.get_event_loop().run_forever()

    except (KeyboardInterrupt, Exception):
        pass

    curses.echo()
    curses.nocbreak()
    curses.endwin()

    sys.exit(0)

