

def predictor_feed(camera_feed):
    while True:
        yield "Christoff", next(camera_feed)

def demo_display(predictor_feed):
    print("Totally Demoing")
