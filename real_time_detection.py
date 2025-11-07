import cv2
from ultralytics import YOLO
import pyttsx3
import threading

# Global variable to hold the TTS engine
engine = None

def speak(text):
    """Function to be run in a separate thread for TTS."""
    global engine
    if engine._inLoop:
        engine.endLoop()
    engine.say(text)
    engine.runAndWait()

def main():
    global engine
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Load the custom YOLOv8 model
    model = YOLO('best.pt')

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tts_thread = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Get the names of detected objects
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = int(box.cls)
                detected_objects.append(model.names[c])

        # Announce the detected objects if the previous announcement is done
        if (tts_thread is None or not tts_thread.is_alive()) and detected_objects:
            unique_objects = set(detected_objects)
            announcement = ", ".join(unique_objects)
            tts_thread = threading.Thread(target=speak, args=(f"I see {announcement}",))
            tts_thread.start()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('Real-time Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # Wait for the last TTS thread to finish
    if tts_thread is not None and tts_thread.is_alive():
        tts_thread.join()

if __name__ == "__main__":
    main()
