1.Imported Libraries: Imported necessary libraries including pandas, numpy, matplotlib, seaborn, and various modules from scikit-learn, nltk, and wordcloud.

2.Loaded Data: Loaded training and test data from CSV files (train.csv and test.csv).

3.Data Exploration and Visualization:

Checked class distribution and addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

4.Data Preprocessing:

Cleaned and tokenized text data using NLTK's tools (word_tokenize for tokenization and PorterStemmer for stemming).
Applied TF-IDF vectorization to convert text data into numerical features suitable for machine learning models.

5.Machine Learning Modeling:

Defined functions to train and evaluate different classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM, XGBoost).
Evaluated models using accuracy, F1-score, and classification reports on both training and validation sets.

6.Model Comparison and Hyperparameter Tuning:

Compared initial model performances before hyperparameter tuning using plots to visualize training and validation accuracies.
Conducted hyperparameter tuning for Logistic Regression, XGBoost, and Linear SVM models to optimize their performance.
Evaluated tuned models and compared their training and validation accuracies.
