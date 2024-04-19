# %%
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

#read in cleaned data
data_for_training = pd.read_csv('cleaned_data.csv', keep_default_na=False)

#Function for additional preprocessing (train-test split, stopword removal, tfidf)

def preprocess(cleaned_df):

    # isolate target variables
    y_combined = cleaned_df[['Job Function', 'Job Role', 'Job Level']]

    # train test split 70/30
    X_train_og, X_test_og, y_combined_train, y_combined_test, ID_train, ID_test = train_test_split(cleaned_df['Title'], y_combined, cleaned_df['Campaign Member ID (18)'], test_size=0.3, random_state=69, stratify=y_combined)

    # split y variables for TRAINING
    y1_train, y2_train, y3_train = y_combined_train.iloc[:, 0], y_combined_train.iloc[:, 1], y_combined_train.iloc[:, 2]
    
    # split y variables for TESTING
    y1_test, y2_test, y3_test = y_combined_test.iloc[:, 0], y_combined_test.iloc[:, 1], y_combined_test.iloc[:, 2]

    # remove stopwords and exclude 'it'
    combined_stop_words = set(stopwords.words('english')).union(set(text.ENGLISH_STOP_WORDS))

    if 'it' in combined_stop_words:
        combined_stop_words.remove('it')
    if 'its' in combined_stop_words:
        combined_stop_words.remove('its')
    if "it's" in combined_stop_words:
        combined_stop_words.remove("it's")
    
    # convert stopwords to list
    my_stopwords_list = list(combined_stop_words)

    #intialize tfidf
    tfidf_vectorizer = TfidfVectorizer(stop_words=my_stopwords_list)

    #fit transform
    X_train = tfidf_vectorizer.fit_transform(X_train_og)
    X_test = tfidf_vectorizer.transform(X_test_og)

    return X_train, X_test, X_test_og, ID_test, y1_train, y2_train, y3_train, y1_test, y2_test, y3_test


# call preprocess to split the data for train/test
X_train, X_test, X_test_og, ID_test, y1_train, y2_train, y3_train, y1_test, y2_test, y3_test = preprocess(data_for_training)

#define param grid
param_grid = {
    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],  
    'fit_prior': [True, False]      
}

#define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

#instantiate cnb for job function
clf1 = ComplementNB()

#train and cross validate
function_cv_scores_cnb = cross_validate(clf1, X_train, y1_train, cv=5, scoring=scoring)

mean_accuracy_scores_func_cnb = function_cv_scores_cnb['test_accuracy'].mean()
mean_precision_scores_func_cnb = function_cv_scores_cnb['test_precision'].mean()
mean_recall_scores_func_cnb = function_cv_scores_cnb['test_recall'].mean()
mean_f1_scores_func_cnb = function_cv_scores_cnb['test_f1'].mean()

print("avg accuracy on train set for job function:", mean_accuracy_scores_func_cnb)
print("avg precision on train set for job function:", mean_precision_scores_func_cnb)
print("avg recall on train set for job function:", mean_recall_scores_func_cnb)
print("avg f1-score on train set for job function:", mean_f1_scores_func_cnb)

#implement gridsearchcv
grid_search_1 = GridSearchCV(estimator=clf1, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_1.fit(X_train, y1_train)

# Get the best hyperparameters
best_params_1 = grid_search_1.best_params_
print("Best hyperparameters for Job Function:", best_params_1)

# Get the best model
best_clf1 = grid_search_1.best_estimator_

# Evaluate the best model on the test set
y1_pred = best_clf1.predict(X_test)
accuracy_1 = accuracy_score(y1_test, y1_pred)
precision_1 = precision_score(y1_test, y1_pred, average='macro')
recall_1 = recall_score(y1_test, y1_pred, average='macro')
f1_1 = f1_score(y1_test, y1_pred, average='macro')

print("Accuracy on test set with best model for Job Function:", accuracy_1)
print("Precision on test set with best model for Job Function:", precision_1)
print("Recall on test set with best model for Job Function:", recall_1)
print("F1-score on test set with best model for Job Function:", f1_1)

#job function confusion matrix
print(confusion_matrix(y1_test, y1_pred))

#job function classification report
print(classification_report(y1_test, y1_pred, digits=4))

#instantiate cnb for job role
clf2 = ComplementNB()

#train and cross validate
role_cv_scores_cnb = cross_validate(clf2, X_train, y2_train, cv=5, scoring=scoring)

mean_accuracy_scores_role_cnb = role_cv_scores_cnb['test_accuracy'].mean()
mean_precision_scores_role_cnb = role_cv_scores_cnb['test_precision'].mean()
mean_recall_scores_role_cnb = role_cv_scores_cnb['test_recall'].mean()
mean_f1_scores_role_cnb = role_cv_scores_cnb['test_f1'].mean()

print("avg accuracy on train set for job role:", mean_accuracy_scores_role_cnb)
print("avg precision on train set for job role:", mean_precision_scores_role_cnb)
print("avg recall on train set for job role:", mean_recall_scores_role_cnb)
print("avg f1-score on train set for job role:", mean_f1_scores_role_cnb)

#implement gridsearchcv
grid_search_2 = GridSearchCV(estimator=clf2, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_2.fit(X_train, y2_train)

# Get the best hyperparameters
best_params_2 = grid_search_2.best_params_
print("Best hyperparameters for Job Role:", best_params_2)

# Get the best model
best_clf2 = grid_search_2.best_estimator_

# Evaluate the best model on the test set
y2_pred = best_clf2.predict(X_test)
accuracy_2 = accuracy_score(y2_test, y2_pred)
precision_2 = precision_score(y2_test, y2_pred, average='macro')
recall_2 = recall_score(y2_test, y2_pred, average='macro')
f1_2 = f1_score(y2_test, y2_pred, average='macro')

print("Accuracy on test set with best model for Job Role:", accuracy_2)
print("Precision on test set with best model for Job Role:", precision_2)
print("Recall on test set with best model for Job Role:", recall_2)
print("F1-score on test set with best model for Job Role:", f1_2)

#job role confusion matrix
print(confusion_matrix(y2_test, y2_pred))

#job role classification report
print(classification_report(y2_test, y2_pred, digits=4))

#instantiate cnb for job level
clf3 = ComplementNB()

#train and cross validate
level_cv_scores_cnb = cross_validate(clf3, X_train, y3_train, cv=5, scoring=scoring)

mean_accuracy_scores_level_cnb = level_cv_scores_cnb['test_accuracy'].mean()
mean_precision_scores_level_cnb = level_cv_scores_cnb['test_precision'].mean()
mean_recall_scores_level_cnb = level_cv_scores_cnb['test_recall'].mean()
mean_f1_scores_level_cnb = level_cv_scores_cnb['test_f1'].mean()

print("avg accuracy on train set for job level:", mean_accuracy_scores_level_cnb)
print("avg precision on train set for job level:", mean_precision_scores_level_cnb)
print("avg recall on train set for job level:", mean_recall_scores_level_cnb)
print("avg f1-score on train set for job level:", mean_f1_scores_level_cnb)

#implement gridsearchcv
grid_search_3 = GridSearchCV(estimator=clf3, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_3.fit(X_train, y3_train)

# Get the best hyperparameters
best_params_3 = grid_search_3.best_params_
print("Best hyperparameters for Job Level:", best_params_3)

# Get the best model
best_clf3 = grid_search_3.best_estimator_

# Evaluate the best model on the test set
y3_pred = best_clf3.predict(X_test)
accuracy_3 = accuracy_score(y3_test, y3_pred)
precision_3 = precision_score(y3_test, y3_pred, average='macro')
recall_3 = recall_score(y3_test, y3_pred, average='macro')
f1_3 = f1_score(y3_test, y3_pred, average='macro')

print("Accuracy on test set with best model for Job Level:", accuracy_3)
print("Precision on test set with best model for Job Level:", precision_3)
print("Recall on test set with best model for Job Level:", recall_3)
print("F1-score on test set with best model for Job Level:", f1_3)

#job level confusion matrix
print(confusion_matrix(y3_test, y3_pred))

#job level classification report
print(classification_report(y3_test, y3_pred, digits=4))

# metrics for all target variables, train and test
cnb_metrics_dict = {
    'Job Function (Train)': [mean_accuracy_scores_func_cnb, mean_precision_scores_func_cnb,mean_recall_scores_func_cnb,mean_f1_scores_func_cnb],
    'Job Function (Test)': [accuracy_1, precision_1, recall_1, f1_1],
    'Job Role (Train)': [mean_accuracy_scores_role_cnb,mean_precision_scores_role_cnb,mean_recall_scores_role_cnb,mean_f1_scores_role_cnb],
    'Job Role (Test)': [accuracy_2, precision_2, recall_2, f1_2],
    'Job Level (Train)': [mean_accuracy_scores_level_cnb,mean_precision_scores_level_cnb,mean_recall_scores_level_cnb,mean_f1_scores_level_cnb],
    'Job Level (Test)': [accuracy_3, precision_3, recall_3, f1_3]
}

cnb_metrics_df = pd.DataFrame(cnb_metrics_dict)

custom_index = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Set the custom index
cnb_metrics_df.index = custom_index

cnb_metrics_df

# create dataframe for predictions
predictions = {'Campaign Member ID (18)': ID_test, \
              'Title': X_test_og,\
              'Predicted Job Function': y1_pred, \
              'Predicted Job Role': y2_pred, \
              'Predicted Job Level': y3_pred, \
              }

predictions_df = pd.DataFrame(predictions)     

#updated role metrics for roles that are under 'IT' but function =! 'IT'
predictions_df.loc[(predictions_df['Predicted Job Role'] != 'NON-ICP') & (predictions_df['Predicted Job Function'] != 'IT'), 'Predicted Job Role'] = 'NON-ICP'

role_pred_fixed = predictions_df['Predicted Job Role']
role_acc_fixed = accuracy_score(y2_test, role_pred_fixed)
role_precision_fixed = precision_score(y2_test, role_pred_fixed, average='macro')
role_recall_fixed = recall_score(y2_test, role_pred_fixed, average='macro')
role_f1_fixed = f1_score(y2_test, role_pred_fixed, average='macro')

print("adjusted accuracy on test set for job role is:", role_acc_fixed)
print("adjusted precision on test set for job role is:", role_precision_fixed)
print("adjusted recall on test set for job role is:", role_recall_fixed)
print("adjusted f1 score on test set for job role is:", role_f1_fixed)


