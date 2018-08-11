import numpy as np
import cv2
import see_a_thing.common as common
import see_a_thing.files as files
import time as time_module

def camera_feed(settings):
    #################################
    # Time in seconds to record     #
    # Frequency in Hz to record in  #
    #################################
    record_until   = time_module.time() + settings["time"]
    sleep_duration = 1 / settings["frequency"]

    print("\n\nCamera Recording For {} Seconds With a Frequency Of {} Hz\n\n"
            .format(settings["time"], settings["frequency"]))

    capture = cv2.VideoCapture(0)
    try:
        while time_module.time() <= record_until:
            ret, frame = capture.read()
            cv2.waitKey(1)

            if settings["display"]:
                cv2.imshow('Camera Feed', frame)

            yield frame

            time_module.sleep(sleep_duration)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


def record(settings):
    feed = camera_feed(settings)

    images = []
    for image in feed:
        images.append(common.preprocess_image(image))

    images = np.array(images)
    files.append_data(settings, images)

