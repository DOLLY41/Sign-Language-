import os
import cv2
import time

# Define the directory to store the collected data
DATA_DIR = './data1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Wait for the user to press 'q' to start collecting data for the current class
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            time.sleep(3)  # Wait for 3 seconds after 'q' key is pressed
            break

    print('Collecting data...')

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

    print('Data collection for class {} complete.'.format(j))

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
