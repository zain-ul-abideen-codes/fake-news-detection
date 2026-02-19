import pandas as pd
import numpy as np
import re
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# # -------------------- Load datasets --------------------
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# Show first 5 rows
print(true.head())
print(fake.head())

# Add label column
true["label"] = 1
fake["label"] = 0

# Combine datasets
news = pd.concat([fake, true], axis=0)

# Check for missing values
print(news.isnull().sum())

# Drop unnecessary columns| Data Cleaning| Sirf text column chahiye model ke liye.
news = news.drop(["title", "subject", "date"], axis=1)
print(news.head())

# Shuffle the dataset
#Data ko mix kar diya. Agar shuffle nahi karte to:Pehle saare fake .Phir saare real news hote. To model ko pata chal jata ki pehle fake news aayegi phir real news. To model bias ho sakta tha. Isliye data ko shuffle kar diya.

news = news.sample(frac=1).reset_index(drop=True)


# #--------------------Function to clean text---------------------
def wordopt(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\n', '', text)  # Remove newlines
    return text

# Apply the wordopt function
news['text'] = news['text'].apply(wordopt)


# Split the dataset into features and labels
x = news['text']
y = news['label']

# Print top 5 for verification
print(x.head())
print(y.head())

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Print shapes of the splits
print("Training set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)

# vectorize the text data using TF-IDF
vectorization = TfidfVectorizer()

xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)



# # ============================================================
# # 1️⃣ Logistic Regression
# # ============================================================

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xv_train, y_train)
# Predict the labels for the test set
pred_lr = lr.predict(xv_test)

# Evaluate the model
lr.score(xv_test, y_test)


# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))

# # ============================================================
# # 2️⃣ Decision Tree
# # ============================================================
# total 4 model use kiye hai. Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier. Sabse pehle Logistic Regression se start kiya. Fir Decision Tree Classifier se start karte hai.
from sklearn.tree import DecisionTreeClassifier
# Create the model
dtc = DecisionTreeClassifier()
# Train the model
dtc.fit(xv_train, y_train)
# Predict the labels for the test set
pred_dtc = dtc.predict(xv_test)
# Evaluate the model
dtc.score(xv_test, y_test)
# Print classification report
print(classification_report(y_test, pred_dtc))



# # ============================================================
# # 3️⃣ Random Forest
# # ============================================================
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(xv_train, y_train)
# Predict the labels for the test set
pred_rfc = rfc.predict(xv_test)
# Evaluate the model
rfc.score(xv_test, y_test)
# Print classification report
print(classification_report(y_test, pred_rfc))



# # ============================================================
# # 4️⃣ Gradient Boosting
# # ============================================================
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(xv_train, y_train)
# Predict the labels for the test set
pred_gbc = gbc.predict(xv_test)
# Evaluate the model
gbc.score(xv_test, y_test)
# Print classification report
print(classification_report(y_test, pred_gbc))


# # ============================================================
# # Function to convert label to human-readable format
# # ============================================================
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"


# # ============================================================
# # Manual Testing Function
# # ============================================================
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format
    (output_lable(pred_lr[0]), output_lable(pred_gbc[0]), output_lable(pred_rfc[0])))


# Manual testing
news = str(input("Enter the news: "))
manual_testing(news)













