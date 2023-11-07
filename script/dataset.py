import pandas as pd
df = pd.read_csv("C:/Users/PC/Desktop/pixeledge/lkm/dataset/inputfile/Dataset Sample.csv")
df['Activity Type'] = df['Activity Type'].str.lower().replace('[^a-z0-9]', '', regex=True)
df['Activity Type'] = df["Activity Type"].str.replace('  ', '', regex=True)
df['Activity Type'] = df["Activity Type"].str.replace(' ', '', regex=True)
df['Emission Resource'] = df['Emission Resource'].str.lower().replace('[^a-z0-9]', '', regex=True)
df['Emission Resource'] = df["Emission Resource"].str.replace('  ', '', regex=True)
df['Emission Resource'] = df["Emission Resource"].str.replace(' ', '', regex=True)
X_train = df[['Activity Type', 'Emission Resource']]
y_train = df['Dataset Name']
# Perform one-hot encoding on the training data
X_train_encoded = pd.get_dummies(X_train, columns=['Activity Type', 'Emission Resource'])
from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Encode the target variable during training
y_train_encoded = label_encoder.fit_transform(y_train)
from sklearn.ensemble import RandomForestClassifier
# Initialize the model
model = RandomForestClassifier(random_state=42)
# Train the model with the encoded target variable
model.fit(X_train_encoded, y_train_encoded)
test_data = pd.read_csv("C:/Users/PC/Desktop/pixeledge/lkm/dataset/inputfile/Test Dataset.csv")
test_data['Activity Type'] = test_data['Activity Type'].str.lower().replace('[^a-z0-9]', '', regex=True)
test_data['Activity Type'] = test_data['Activity Type'].str.replace(' ', '', regex=True)
test_data['Activity Type'] = test_data['Activity Type'].str.replace('  ', '', regex=True)
test_data['Emission Resource'] = test_data['Emission Resource'].str.lower().replace('[^a-z0-9]', '', regex=True)
test_data['Emission Resource'] = test_data['Emission Resource'].str.replace(' ', '', regex=True)
test_data['Emission Resource'] = test_data['Emission Resource'].str.replace('  ', '', regex=True)
# Separate features in test data (X_test)
X_test = test_data[['Activity Type', 'Emission Resource']]
# Perform one-hot encoding on the test data using the same columns used for training
X_test_encoded = pd.get_dummies(X_test, columns=['Activity Type', 'Emission Resource'])
# Ensure columns is in same sequence to avoid error of unseen value
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]
# Use the trained model to make predictions on the preprocessed test data
predictions_encoded = model.predict(X_test_encoded)
# Decode the predictions back to the original labels
predictions_decoded = label_encoder.inverse_transform(predictions_encoded)
# Assign the decoded predictions to a Dataset Name column
test_data['Dataset Name'] = predictions_decoded
test_data.to_csv('C:/Users/PC/Desktop/pixeledge/lkm/dataset/outputfile/test_dataset.csv', index=False)
