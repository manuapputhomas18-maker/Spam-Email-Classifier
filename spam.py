
# SPAM CLASSIFIER USING MACHINE LEARNING
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# CREATE DATASET
# ==========================================

messages = np.array([
    "Win a free iPhone now",
    "Congratulations you won a lottery",
    "Claim your cash reward today",
    "Free recharge available",
    "Exclusive offer just for you",
    "Call me when you reach home",
    "Meeting starts at 10 AM",
    "Project submission tomorrow",
    "Please send the report",
    "Are you available today",
    "Get a loan instantly",
    "You have won 100000 rupees",
    "Urgent! Claim your prize",
    "Let's meet in the office",
    "Your order has been delivered",
    "Dinner at 8 PM"
])

labels = np.array([
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham"
])

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({
    "Message": messages,
    "Label": labels
})

print("\nDataset:")
print(df)

# ==========================================
# FEATURE EXTRACTION
# ==========================================

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["Message"])
y = df["Label"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

# ==========================================
# TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# NUMPY OPERATIONS
# ==========================================

print("\nNumPy Operations:")

print("Total Messages:", np.size(messages))
print("Spam Messages:", np.sum(labels == "spam"))
print("Ham Messages:", np.sum(labels == "ham"))

# ==========================================
# USER INPUT PREDICTION
# ==========================================

while True:

    msg = input("\nEnter a Message: ")

    msg_vector = vectorizer.transform([msg])

    result = model.predict(msg_vector)

    print("Prediction:", result[0].upper())

    choice = input("Do you want to continue? (yes/no): ")

    if choice.lower() != "yes":
        break

print("\nProgram Finished")