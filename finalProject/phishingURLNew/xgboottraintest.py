import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle  # To save the model

# Load the dataset
data = pd.read_csv('phishing.csv')

# Split features and target
X = data.drop('class', axis=1)  # Replace 'target' with the actual target column name
y = data['class']

# Ensure target values are in [0, 1]
print(f"Unique values in target: {y.unique()}")  # Debug step to check target values
if not set(y.unique()).issubset({0, 1}):
    y = y.map({-1: 0, 1: 1})  # Adjust this mapping as needed

# Ensure target is an integer
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',  # For binary classification
    'max_depth': 6,                  # Maximum depth of trees
    'eta': 0.3,                      # Learning rate
    'eval_metric': 'logloss'         # Evaluation metric
}

# Train the model
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 100  # Number of boosting rounds
model = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

# Make predictions
y_pred_prob = model.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save the model as model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")
