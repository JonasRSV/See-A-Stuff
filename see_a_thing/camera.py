import cv2
import see_a_thing.common as common
import time as time_module

def camera_feed(time, frequency, display=True):
    #################################
    # Time in seconds to record     #
    # Frequency in Hz to record in  #
    #################################
    record_until   = time_module.time() + time
    sleep_duration = 1 / frequency

    capture = cv2.VideoCapture(0)
    try:
        while time_module.time() <= record_until:
            ret, frame = capture.read()
            cv2.waitKey(1)

            if display:
                cv2.imshow('frame', frame)

            yield frame

            time_module.sleep(sleep_duration)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()

