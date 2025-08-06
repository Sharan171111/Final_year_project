import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Load the dataset
df = pd.read_csv('phishing.csv')

# Assuming 'label' is the target column and the rest are features
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (SVM often benefits from feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the SVM model
model = SVC(kernel='linear', random_state=42)  # You can experiment with different kernels, e.g., 'rbf', 'poly'

# Train the model
model.fit(X_train_scaled, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))




# Save the trained model to a file using pickle
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler as well
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model saved as svm_model.pkl and scaler saved as scaler.pkl")


# Load the saved model from the file
with open('svm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the saved scaler from the file
with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)


