import os  # Importing the os module for file and directory operations
import cv2  # Importing OpenCV library for video capture and image processing

# Define the directory to store the collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Check if the directory doesn't exist
    os.makedirs(DATA_DIR)  # Create the directory if it doesn't exist

number_of_classes = 3  # Define the number of classes or categories for data collection
dataset_size = 100  # Define the number of samples to collect for each class

# Open a video capture device (webcam) with index 0
cap = cv2.VideoCapture(0)

# Loop over each class for data collection
for j in range(number_of_classes):
    # Create a subdirectory for each class inside the main data directory
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Display a message indicating the class for which data is being collected
    print('Collecting data for class {}'.format(j))

    # Wait for the user to press 'q' to start collecting data for the current class
    while True:
        ret, frame = cap.read()  # Read a frame from the video capture device
        # Display a message prompting the user to press 'q' to start
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Display the frame
        # Wait for 25 milliseconds for a key press event, and check if 'q' is pressed to break out of the loop
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0  # Initialize a counter to keep track of the number of samples collected
    # Loop to capture the specified number of samples for the current class
    while counter < dataset_size:
        ret, frame = cap.read()  # Read a frame from the video capture device
        cv2.imshow('frame', frame)  # Display the frame
        cv2.waitKey(25)  # Wait for 25 milliseconds
        # Save the frame as an image file with a unique name in the appropriate class subdirectory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment the counter for the next iteration

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
