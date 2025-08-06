import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('phishing_data.csv')
urls = data['url']
labels = data['type']
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(urls)
adj_matrix = (features * features.T).A
graph = nx.from_numpy_matrix(adj_matrix)
train_graph, test_graph, train_labels, test_labels = train_test_split(graph, labels, test_size=0.2, random_state=42)
k = 3  # Number of nearest neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(train_graph, train_labels)
predictions = knn_classifier.predict(test_graph)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
