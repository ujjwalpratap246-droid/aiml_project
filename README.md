# SEMESTER-1-AIML-PROJECT
SPAM MESSAGE CLASSIFICATION USING PYTHON AND SCIKIT-LEARN

#RUNNING CODE:-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_spam_model():
    print("--- Loading Dataset ---")
    try:
        # Load the uploaded dataset
        df = pd.read_csv('Spam_SMS.csv')
        print(f"Successfully loaded {len(df)} messages.")
    except FileNotFoundError:
        print("Error: 'Spam_SMS.csv' not found. Please make sure the file is in the same directory.")
        return

    # 1. Data Preprocessing
    # Check if columns exist as expected from your file snippet
    if 'Class' not in df.columns or 'Message' not in df.columns:
        print("Error: Dataset must contain 'Class' and 'Message' columns.")
        print(f"Found columns: {df.columns}")
        return

    # Convert labels to numbers: ham -> 0, spam -> 1
    # This is necessary because machine learning models work with numbers, not strings.
    
    df['label_num'] = df['Class'].map({'ham': 0, 'spam': 1})

    # Define features (X) and target (y)
    X = df['Message']
    y = df['label_num']

    # 2. Split the data
    # We split into 80% training data and 20% testing data.
    # stratify=y ensures we have the same ratio of spam/ham in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining on {len(X_train)} messages.")
    print(f"Testing on {len(X_test)} messages.")

    # 3. Feature Extraction (Vectorization)
    # We use TF-IDF (Term Frequency-Inverse Document Frequency).
    # It converts text into numbers, giving less weight to common words (like "the", "is")
    # and more weight to unique, meaningful words.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    
    # Learn vocabulary from training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Transform test data using the *same* vocabulary
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Model Training
    # Multinomial Naive Bayes is the standard baseline for text classification.
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("\nModel training complete.")

    # 5. Evaluation
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Let's test the model with some custom examples
    print("-" * 30)
    print("Interactive Test:")
    
    custom_messages = [
        "Congrats! You've won a $1000 gift card. Call now to claim.",
        "Hey, are we still meeting for dinner tonight?",
        "URGENT! Your account is locked. Click here to reset password."
    ]
    
    # Transform these new messages using the already trained vectorizer
    custom_vec = vectorizer.transform(custom_messages)
    predictions = model.predict(custom_vec)
    
    for msg, label in zip(custom_messages, predictions):
        result = "SPAM" if label == 1 else "HAM"
        print(f"\nMessage: {msg}\nPrediction: {result}")

if __name__ == "__main__":
    train_spam_model()
 
