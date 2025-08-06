🛡️ Phishing Website Detection using XGBoost
This project detects phishing websites using machine learning, specifically the XGBoost algorithm. The model is trained on URL-based features to predict whether a given URL is legitimate or phishing.

🚀 Features
✅ URL feature extraction (e.g., length, special characters, domain info)

✅ Trained XGBoost classification model

✅ Web interface built with Flask for real-time URL predictions

✅ Error handling and logging support

✅ Modular code structure for easy maintenance and extension



🧠 Model Info
Algorithm: XGBoost Classifier

Features Used: 30+ hand-engineered features from URLs

Target Variable: Phishing (1) or Legitimate (0)

⚙️ Setup Instructions
🔽 1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/yourusername/phishing-website-xgboost.git
cd phishingURLNew
🐍 2. Create & Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
📦 3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
🏃‍♂️ 4. Run the Flask App
bash
Copy
Edit
python app.py
Then open your browser and visit:
http://127.0.0.1:5000

🧪 Example
Enter a URL like:

pgsql
Copy
Edit
http://paypal.login.verify-user-session.com
Prediction Output:

pgsql
Copy
Edit
⚠️ Warning! This is a Phishing Website.
📊 Dataset (Optional)
If you want to retrain the model:

Use a labeled dataset of URLs (phishing and legitimate)

Extract features using feature.py

Train the model using XGBoost and save it using joblib or pickle

📌 Requirements
Python 3.8+

Flask

XGBoost

Pandas

Numpy

Scikit-learn

(See requirements.txt for exact versions)

🤝 Contributing
Feel free to fork the repo and submit a pull request with improvements. All contributions are welcome!
