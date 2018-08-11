import tkinter as tk
import see_a_thing.graphs as graphs
import see_a_thing.common as common
import see_a_thing.camera as camera
import json
import asyncio
import websockets
import tensorflow as tf
import curses
import sys


def predictions(camera_feed, graph):
    
    session = tf.get_default_session()
    inputs, category, probs, categories = graph

    categories = common.decode_bytes(session.run(categories))
    try:
        while True:
            cam_image = next(camera_feed)

            image = common.preprocess_image(cam_image)

            p, c = session.run((probs,
                                category),
                               feed_dict={inputs: [image]})
                                      

            p = p[0]
            c = common.decode_bytes(c)[0]

            yield c, categories, p

    except StopIteration:
        sys.stderr.write("\n Camera feed done, qutting predictor feed.")


def monitor(settings):
    camera_feed = camera.camera_feed(settings)
    stdscr      = curses.initscr()

    curses.noecho()
    curses.cbreak()

    with tf.Session() as session:
        graph = graphs.GraphBuilder.restore_graph(settings["model_path"])
        predictions_feed = predictions(camera_feed, graph)

        try:
            while True:
                c, cs, ps = next(predictions_feed)

                stdscr.addstr(0, 0, "{} is in front of me".format(c))
                for i, (cat, prob) in enumerate(zip(cs, ps)):
                    stdscr.addstr(i + 1, 0, "{} {}%".format(cat, round(prob * 100, 2)))
                stdscr.refresh()
        except StopIteration:
            sys.stderr.write("\n Predictor feed done, qutting Demo feed.")

        curses.echo()
        curses.nocbreak()
        curses.endwin()


def serve(settings):
    camera_feed = camera.camera_feed(settings)

    with tf.Session() as session:
        graph = graphs.GraphBuilder.restore_graph(settings["model_path"])
        predictions_feed = predictions(camera_feed, graph)

        message = {"category": None,
                   "categories": [],
                   "probabilites": []}

        @asyncio.coroutine
        def pred_server(websocket, path):
            while True:
                print("Message", message)
                yield from websocket.send(json.dumps(message))
                yield from asyncio.sleep(1)

        @asyncio.coroutine
        def feed_update():
            while True:
                c, cs, ps = next(predictions_feed)
                message["category"]     = str(c)
                message["categories"]   = list(cs)
                message["probabilites"] = list(map(lambda f: float(f), ps))

                yield from asyncio.sleep(0.01)

        server = websockets.serve(pred_server, "0.0.0.0", 5000)

        asyncio.get_event_loop().run_until_complete(server)
        asyncio.async(feed_update())

        print("Serving at 0.0.0.0:5000")
        asyncio.get_event_loop().run_forever()





