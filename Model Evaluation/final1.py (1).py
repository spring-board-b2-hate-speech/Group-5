#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


# # STEP 1:DATA LOADING AND PREPROCESSING 

# In[3]:


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    return text.strip()  # Strip leading/trailing whitespace


# In[7]:


try:
    df = pd.read_csv(r"C:\Users\sneha\Downloads\data2.csv")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)


# In[8]:


if 'Class' in df.columns:
    df.drop(columns=['Class'], inplace=True)


# In[9]:


text_data = df['tweet'].tolist()
labels = df['class'].tolist()


# In[10]:


preprocessor = FunctionTransformer(lambda x: [preprocess_text(text) for text in x])


# # STEP2:SAVING TF-IDF VECTORIZER

# In[11]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000) 


# In[12]:


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('tfidf', tfidf_vectorizer)
])


# In[13]:


# Fit-transform the pipeline on the text data
try:
    tfidf_matrix = pipeline.fit_transform(text_data)
except Exception as e:
    print(f"Error in pipeline fit-transform: {e}")
    exit(1)


# In[14]:


tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


# In[15]:


output_file = 'tfidf_data.csv'


# In[16]:


tfidf_df.to_csv(output_file, index=False)
print(f"TF-IDF vectorized data saved to {output_file}")


# # STEP 3: DATA SPLITTING
# Data is splitted into 3 steps:
# Train,test and validation in the ratio 70:15:15

# In[ ]:


# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(tfidf_matrix, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# # STEP 4: Model Training and Testing

# In[18]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


# # 1.Random Forest

# In[19]:


from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Predict on validation set
val_predictions_rf = rf_classifier.predict(X_val)


# In[20]:


# Evaluate performance on validation set
val_accuracy_rf = accuracy_score(y_val, val_predictions_rf)
print(f'Validation Accuracy: {val_accuracy_rf:.4f}')

val_classification_report_rf = classification_report(y_val, val_predictions_rf)
print('Classification Report on Validation Set:')
print(val_classification_report_rf)

# Confusion matrix for validation set
val_conf_matrix_rf = confusion_matrix(y_val, val_predictions_rf)
print('Confusion Matrix on Validation Set:')
print(val_conf_matrix_rf)


# In[21]:


# Optionally, evaluate on test set if needed
test_predictions_rf = rf_classifier.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, test_predictions_rf)
print(f'Test Accuracy: {test_accuracy_rf:.4f}')

test_classification_report_rf = classification_report(y_test, test_predictions_rf)
print('Classification Report on Test Set:')
print(test_classification_report_rf)

# Confusion matrix for test set
test_conf_matrix_rf = confusion_matrix(y_test, test_predictions_rf)
print('Confusion Matrix on Test Set:')
print(test_conf_matrix_rf)


# # 2.LOGISTIC REGRESSION

# In[22]:


from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression classifier
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)

# Train Logistic Regression classifier
lr_classifier.fit(X_train, y_train)

# Predict on validation set
val_predictions_lr = lr_classifier.predict(X_val)


# In[23]:


# Evaluate performance on validation set
val_accuracy_lr = accuracy_score(y_val, val_predictions_lr)
print(f'Validation Accuracy (Logistic Regression): {val_accuracy_lr:.4f}')

val_classification_report_lr = classification_report(y_val, val_predictions_lr)
print('Classification Report on Validation Set (Logistic Regression):')
print(val_classification_report_lr)

# Confusion matrix for validation set
val_conf_matrix_lr = confusion_matrix(y_val, val_predictions_lr)
print('Confusion Matrix on Validation Set (Logistic Regression):')
print(val_conf_matrix_lr)


# In[24]:


# Optionally, evaluate on test set if needed
test_predictions_lr = lr_classifier.predict(X_test)
test_accuracy_lr = accuracy_score(y_test, test_predictions_lr)
print(f'Test Accuracy (Logistic Regression): {test_accuracy_lr:.4f}')

test_classification_report_lr = classification_report(y_test, test_predictions_lr)
print('Classification Report on Test Set (Logistic Regression):')
print(test_classification_report_lr)

# Confusion matrix for test set
test_conf_matrix_lr = confusion_matrix(y_test, test_predictions_lr)
print('Confusion Matrix on Test Set (Logistic Regression):')
print(test_conf_matrix_lr)


# # 3. XGBOOST

# In[26]:


get_ipython().system('pip install xgboost')


# In[27]:


# Step 1: Install XGBoost library (uncomment the following line if running in a local environment)
# !pip install xgboost

# Step 2: Import the XGBoost library
from xgboost import XGBClassifier

# Assuming X_train, y_train, X_val are already defined
# If not, ensure you have your training and validation data prepared

# Step 3: Initialize XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)

# Step 4: Train XGBoost classifier
xgb_classifier.fit(X_train, y_train)

# Step 5: Predict on validation set
val_predictions_xgb = xgb_classifier.predict(X_val)

# Optional: Print predictions or evaluate the model
print(val_predictions_xgb)


# In[28]:


# Evaluate performance on validation set
val_accuracy_xgb = accuracy_score(y_val, val_predictions_xgb)
print(f'Validation Accuracy (XGBoost): {val_accuracy_xgb:.4f}')

val_classification_report_xgb = classification_report(y_val, val_predictions_xgb)
print('Classification Report on Validation Set (XGBoost):')
print(val_classification_report_xgb)

# Confusion matrix for validation set
val_conf_matrix_xgb = confusion_matrix(y_val, val_predictions_xgb)
print('Confusion Matrix on Validation Set (XGBoost):')
print(val_conf_matrix_xgb)


# In[29]:


get_ipython().system('pip install joblib')
import joblib

joblib.dump(xgb_classifier, 'xgb_classifier.joblib') # Save the trained model using joblib


# In[30]:


import pickle

# Assuming 'tfidf_vectorizer' is your fitted TfidfVectorizer object
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)  # Save the vectorizer to a file

# Now you can load it
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)


# In[31]:


# Optionally, evaluate on test set if needed
test_predictions_xgb = xgb_classifier.predict(X_test)
test_accuracy_xgb = accuracy_score(y_test, test_predictions_xgb)
print(f'Test Accuracy (XGBoost): {test_accuracy_xgb:.4f}')

test_classification_report_xgb = classification_report(y_test, test_predictions_xgb)
print('Classification Report on Test Set (XGBoost):')
print(test_classification_report_xgb)

# Confusion matrix for test set
test_conf_matrix_xgb = confusion_matrix(y_test, test_predictions_xgb)
print('Confusion Matrix on Test Set (XGBoost):')
print(test_conf_matrix_xgb)


# # 4. SVM

# In[32]:


from sklearn.svm import SVC

# Initialize SVM classifier
svm_classifier = SVC(random_state=42)

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict on validation set
val_predictions_svm = svm_classifier.predict(X_val)


# In[33]:


# Evaluate performance on validation set
val_accuracy_svm = accuracy_score(y_val, val_predictions_svm)
print(f'Validation Accuracy (SVM): {val_accuracy_svm:.4f}')

val_classification_report_svm = classification_report(y_val, val_predictions_svm)
print('Classification Report on Validation Set (SVM):')
print(val_classification_report_svm)

# Confusion matrix for validation set
val_conf_matrix_svm = confusion_matrix(y_val, val_predictions_svm)
print('Confusion Matrix on Validation Set (SVM):')
print(val_conf_matrix_svm)


# In[34]:


# Optionally, evaluate on test set if needed
test_predictions_svm = svm_classifier.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, test_predictions_svm)
print(f'Test Accuracy (SVM): {test_accuracy_svm:.4f}')

test_classification_report_svm = classification_report(y_test, test_predictions_svm)
print('Classification Report on Test Set (SVM):')
print(test_classification_report_svm)

# Confusion matrix for test set
test_conf_matrix_svm = confusion_matrix(y_test, test_predictions_svm)
print('Confusion Matrix on Test Set (SVM):')
print(test_conf_matrix_svm)


# # 5.DECISION TREE

# In[35]:


from sklearn.tree import DecisionTreeClassifier

# Initialize Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Predict on validation set
val_predictions_dt = dt_classifier.predict(X_val)


# In[36]:


# Evaluate performance on validation set
val_accuracy_dt = accuracy_score(y_val, val_predictions_dt)
print(f'Validation Accuracy (Decision Tree): {val_accuracy_dt:.4f}')

val_classification_report_dt = classification_report(y_val, val_predictions_dt)
print('Classification Report on Validation Set (Decision Tree):')
print(val_classification_report_dt)

# Confusion matrix for validation set
val_conf_matrix_dt = confusion_matrix(y_val, val_predictions_dt)
print('Confusion Matrix on Validation Set (Decision Tree):')
print(val_conf_matrix_dt)


# In[37]:


# Optionally, evaluate on test set if needed
test_predictions_dt = dt_classifier.predict(X_test)
test_accuracy_dt = accuracy_score(y_test, test_predictions_dt)
print(f'Test Accuracy (Decision Tree): {test_accuracy_dt:.4f}')

test_classification_report_dt = classification_report(y_test, test_predictions_dt)
print('Classification Report on Test Set (Decision Tree):')
print(test_classification_report_dt)

# Confusion matrix for test set
test_conf_matrix_dt = confusion_matrix(y_test, test_predictions_dt)
print('Confusion Matrix on Test Set (Decision Tree):')
print(test_conf_matrix_dt)


# # STEP 5: PLOTTING 

# In[38]:


import matplotlib.pyplot as plt
import numpy as np


# In[39]:


# Test accuracies for each classifier
classifiers = ['Random Forest', 'Logistic Regression', 'Decision Tree', 'XGBoost', 'SVM']
accuracies = [test_accuracy_rf, test_accuracy_lr, test_accuracy_dt, test_accuracy_xgb, test_accuracy_svm]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])

# Adding labels and title
plt.xlabel('Classifiers')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Classifiers')
plt.ylim([0.0, 1.0])  # Setting y-axis limit to better visualize differences
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Adding the actual value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom')

# Display the plot
plt.tight_layout()
plt.show()


# # STEP 6:PREDICTIONS

# In[ ]:


import pickle
import joblib
import numpy as np

# Load the model and vectorizer from the file
loaded_model = joblib.load("xgb_classifier.joblib")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)

class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

def classify_text(text):
    # Transform the input data using the loaded TF-IDF vectorizer
    input_transformed = loaded_tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = loaded_model.predict(input_transformed)

    # Get the predicted class label
    predicted_class = class_labels[prediction[0]]
    return predicted_class

if __name__ == "__main__":
    while True:
        input_text = input("Enter text to classify: ")

        if input_text.lower() == 'exit':
            print("Exiting...")
            break

        # Classify the input text
        predicted_class = classify_text(input_text)

        # Print the classification result
        print(f"Input: {input_text} -> Prediction: {predicted_class}")


# In[ ]:


import os

print("Current Working Directory:", os.getcwd())
print("Is 'xgb_classifier.joblib' in CWD?", os.path.exists('xgb_classifier.joblib'))
print("Is 'tfidf_vectorizer.pkl' in CWD?", os.path.exists('tfidf_vectorizer.pkl'))


# In[ ]:




