# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('dataset/Dataset Sample (1) (3).csv')

# Split the dataset into features (X) and target (y)
X = data[['Activity Type', 'Emission Resource']]
y = data['Dataset Name']

# One-hot encode the categorical features
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# You can now use this trained model to make predictions on new data
# For example, to predict the 'Dataset Name' for a new data point:
# new_data_point = [['Activity_Type_Value', 'Emission_Resource_Value']]
# new_data_point_encoded = encoder.transform(new_data_point)
# prediction = rf_classifier.predict(new_data_point_encoded)
# print(f"Predicted Dataset Name: {prediction[0]}")
new_data = pd.read_csv('dataset/Test Dataset (1).csv')

# Extract the 'Activity Type' and 'Emission Resource' columns
X_new = new_data[['Activity Type', 'Emission Resource']]

# Use the same encoder that was used during training to perform one-hot encoding
X_new_encoded = encoder.transform(X_new)

# Use the trained Random Forest model to make predictions for 'Dataset Name'
predicted_dataset_names = rf_classifier.predict(X_new_encoded)

# Add the predicted 'Dataset Name' values to the new data
new_data['Dataset Name'] = predicted_dataset_names

# Save the new data with the predicted 'Dataset Name' values to a new CSV file
new_data.to_csv('dataset/output.csv', index=False)
