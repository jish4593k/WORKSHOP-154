import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tkinter import filedialog
import tkinter as tk

def preprocess_text_data(text_data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_data)
    return X

def k_means_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels


def visualize_clusters(X, labels):
    data = pd.DataFrame(X.toarray())
    data['Cluster'] = labels
    sns.pairplot(data, hue='Cluster')
    plt.show()

def build_and_train_neural_network(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    return model

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        return file_path
    ret
def process_data_and_train_model(file_path):
    # Load data from the CSV file
    df = pd.read_csv(file_path)
    X = preprocess_text_data(df['Tweet'].values)
    
   
    n_clusters = 2  # You can adjust the number of clusters
    labels = k_means_clustering(X, n_clusters)
    

    visualize_clusters(X, labels)
    

    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    input_dim = X_train.shape[1]
    model = build_and_train_neural_network(X_train, y_train, input_dim)
 
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Neural Network Accuracy: {accuracy}")


def main():
    root = tk.Tk()
    root.title("Text Data Analysis")
    root.geometry("300x100")

    select_button = tk.Button(root, text="Select CSV File", command=lambda: process_data_and_train_model(select_file()))
    select_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
