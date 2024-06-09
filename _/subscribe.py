import tellopy
import cv2
import av
import numpy as np

def flight_data_handler(event, sender, data, **args):
    if event == sender.EVENT_FLIGHT_DATA:
        print(data)

def video_frame_handler(event, sender, data, **args):
    if event == sender.EVENT_VIDEO_FRAME:
        try:
            # Initialize PyAV container from bytes
            container = av.open(data, format='h264')
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='bgr24')
                cv2.imshow('Tello Video Stream', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except av.AVError as e:
            print(f"Error decoding video frame: {e}")

def main():
    tello = tellopy.Tello()

    try:
        tello.connect()
        tello.wait_for_connection(60.0)
        tello.start_video()

        # Subscribe to flight data and video frame events
        tello.subscribe(tello.EVENT_FLIGHT_DATA, flight_data_handler)
        tello.subscribe(tello.EVENT_VIDEO_FRAME, video_frame_handler)

        cv2.namedWindow('Tello Video Stream', cv2.WINDOW_NORMAL)

        while True:
            # Keep the program running to receive events
            pass

    except Exception as ex:
        print(ex)

    finally:
        tello.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
