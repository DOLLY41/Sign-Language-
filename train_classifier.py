# import pickle  # Import the pickle module for loading and saving data
# import numpy as np  # Import the NumPy library for numerical operations
# from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier from scikit-learn
# from sklearn.model_selection import train_test_split  # Import train_test_split function for splitting the data
# from sklearn.metrics import accuracy_score  # Import accuracy_score function for evaluating model performance

# # Load the dataset containing hand landmark data and labels from the 'data.pickle' file
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = data_dict['data']  # Extract hand landmark data
# labels = data_dict['labels']  # Extract corresponding labels

# # Calculate the maximum number of features for each hand in the dataset
# max_features_per_hand = max(len(hand_features) for hand_features in data)

# # Pad or truncate features for each hand to have the same length as the maximum number of features
# data_padded = []
# for hand_features in data:
#     padded_features = hand_features + [0] * (max_features_per_hand - len(hand_features))
#     data_padded.append(padded_features)
# data = np.array(data_padded)  # Convert the data to a NumPy array for compatibility
# labels = np.array(labels)  # Convert the labels to a NumPy array for compatibility

# # Split the dataset into training and testing subsets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Initialize a RandomForestClassifier model
# model = RandomForestClassifier()

# # Train the RandomForestClassifier model on the training data
# model.fit(x_train, y_train)

# # Make predictions on the test data
# y_predict = model.predict(x_test)

# # Calculate the accuracy of the model
# score = accuracy_score(y_predict, y_test)
# print("Number of features in x_test:", x_test.shape[1])

# # Print the accuracy score
# print('{}% of samples were classified correctly !'.format(score * 100))

# # Save the trained model to a file using pickle
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset containing hand landmark data and labels from the 'data.pickle' file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']  # Extract hand landmark data
labels = data_dict['labels']  # Extract corresponding labels

# Truncate or pad features for each hand to have exactly 42 features
desired_length_per_hand = 21
data_modified = []
for hand_features in data:
    if len(hand_features) < desired_length_per_hand * 2:
        padded_features = hand_features + [0] * ((desired_length_per_hand * 2) - len(hand_features))
    elif len(hand_features) > desired_length_per_hand * 2:
        padded_features = hand_features[:desired_length_per_hand * 2]
    else:
        padded_features = hand_features
    data_modified.append(padded_features)
data = np.array(data_modified)  # Convert the data to a NumPy array for compatibility
labels = np.array(labels)  # Convert the labels to a NumPy array for compatibility

# Split the dataset into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the RandomForestClassifier model on the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print("Number of features in x_test:", x_test.shape[1])
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
