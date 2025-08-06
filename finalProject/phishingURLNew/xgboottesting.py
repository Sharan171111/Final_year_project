import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('phishing.csv')

# Convert non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
for col in non_numeric_columns:
    data[col] = data[col].astype('category').cat.codes

# Split features and target
X = data.drop('class', axis=1)  # Replace 'target' with the actual target column name
y = data['class']

# Ensure target values are in [0, 1]
if not set(y.unique()).issubset({0, 1}):
    y = y.map({-1: 0, 1: 1})

y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.3,
    'eval_metric': 'logloss'
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Save the model as model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# -------------- Test the model on new data ----------------------

# Load the model from model.pkl
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare sample data for prediction
# Replace 'sample_data.csv' with your actual sample data file
sample_data = pd.read_csv('new_data.csv')

# Convert non-numeric columns in the sample data
sample_non_numeric_columns = sample_data.select_dtypes(include=['object']).columns
for col in sample_non_numeric_columns:
    sample_data[col] = sample_data[col].astype('category').cat.codes

# Convert the sample data to DMatrix for prediction
dsample = xgb.DMatrix(sample_data)

# Predict probabilities for the sample data
sample_predictions_prob = model.predict(dsample)

# Convert probabilities to binary labels (1 for phishing, 0 for legitimate)
sample_predictions = [1 if prob > 0.5 else 0 for prob in sample_predictions_prob]

# Output the predictions
print("Sample Predictions (0 = Legitimate, 1 = Phishing):")
print(sample_predictions)
