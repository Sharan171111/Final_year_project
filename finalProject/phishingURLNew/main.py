import re
import urllib.parse
import whois
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Prepare the dataset of legitimate and phishing URLs
legitimate_urls = ["https://www.google.com", "https://www.amazon.com"]
phishing_urls = ["http://www.example.com", "http://www.phishing.com"]


# Feature extraction function
def extract_features(url):
    features = []

    # Check if the URL uses HTTPS
    features.append(int(url.startswith("https")))

    # Extract domain and path from the URL
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # Check if the domain is registered for a longer period (Phishing websites are often short-lived)
    try:
        w = whois.whois(domain)
        features.append(int((w.expiration_date - w.creation_date).days >= 365))
    except:
        features.append(0)

    # Check the length of the URL
    features.append(len(url))

    # Check if the URL contains '@' symbol (usually found in phishing URLs)
    features.append(int("@" in url))

    # Check if the URL's domain is an IP address (usually found in phishing URLs)
    features.append(int(bool(re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain))))

    # Check if the URL redirects to another URL
    try:
        response = requests.get(url)
        features.append(int(url != response.url))
    except:
        features.append(0)

    # Check the number of dots in the domain name
    features.append(domain.count('.'))

    # Check the length of the domain name
    features.append(len(domain))

    # Check the length of the path
    features.append(len(path))

    return features


# Prepare the feature vectors and labels
all_urls = legitimate_urls + phishing_urls
labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
feature_vectors = vectorizer.fit_transform([extract_features(url) for url in all_urls])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Detect phishing URLs
new_urls = ["https://www.openai-phishing.com", "https://www.yahoo.com"]
new_feature_vectors = vectorizer.transform([extract_features(url) for url in new_urls])
predictions = clf.predict(new_feature_vectors)

for url, prediction in zip(new_urls, predictions):
    if prediction == 1:
        print(f"Phishing URL: {url}")
    else:
        print(f"Legitimate URL: {url}")
