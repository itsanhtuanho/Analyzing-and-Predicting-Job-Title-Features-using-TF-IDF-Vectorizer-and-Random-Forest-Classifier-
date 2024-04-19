
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import cross_validate
from imblearn.ensemble import EasyEnsembleClassifier

data_for_training = pd.read_csv('cleaned_data.csv', keep_default_na=False)

#Function for additional preprocessing (train-test split, stopword removal, tfidf)
def preprocess(cleaned_df):
    # isolate target variables
    y_combined = cleaned_df[['Job Function', 'Job Role', 'Job Level']]
    # train test split 70/30
    X_train_og, X_test_og, y_combined_train, y_combined_test, ID_train, ID_test = train_test_split(cleaned_df['Title'], y_combined, cleaned_df['Campaign Member ID (18)'], test_size=0.3, random_state=69, stratify=y_combined)
    # split y variables for TRAINING and TESTING
    y1_train, y2_train, y3_train = y_combined_train.iloc[:, 0], y_combined_train.iloc[:, 1], y_combined_train.iloc[:, 2]
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


X_train, X_test, X_test_og, ID_test, y1_train, y2_train, y3_train, y1_test, y2_test, y3_test = preprocess(data_for_training)

# Modeling
rf = RandomForestClassifier(n_estimators=100, random_state=69)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

## Function (remove verbose=2 if user does not want to keep track of model training progress)
ee_rf_func = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
    # Cross-validation
cv_scores_func_ee = cross_validate(ee_rf_func, X_train, y1_train, cv=5, scoring=scoring)
    # Getting averaged metrics from cross-validation
mean_accuracy_func_ee = cv_scores_func_ee['test_accuracy'].mean()
mean_precision_func_ee = cv_scores_func_ee['test_precision'].mean()
mean_recall_func_ee = cv_scores_func_ee['test_recall'].mean()
mean_f1_func_ee = cv_scores_func_ee['test_f1'].mean()
    # Fit on training data
ee_rf_function_classifier = ee_rf_func.fit(X_train, y1_train)
    # Prediction
ypred_test_func_ee = ee_rf_function_classifier.predict(X_test)
    # Metric on Testing data
test_accuracy_func_ee = accuracy_score(y1_test, ypred_test_func_ee)
test_precision_func_ee = precision_score(y1_test, ypred_test_func_ee, average='macro')
test_recall_func_ee = recall_score(y1_test, ypred_test_func_ee, average='macro')
test_f1_func_ee = f1_score(y1_test, ypred_test_func_ee, average='macro')
    # Classification Report to see break down of predictions
function_cr = classification_report(y1_test, ypred_test_func_ee, digits=4)


## Role
ee_rf_role = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
    # Cross-validation
cv_scores_role_ee = cross_validate(ee_rf_role, X_train, y2_train, cv=5, scoring=scoring)
    # Getting averaged metrics from cross-validation
mean_accuracy_role_ee = cv_scores_role_ee['test_accuracy'].mean()
mean_precision_role_ee = cv_scores_role_ee['test_precision'].mean()
mean_recall_role_ee = cv_scores_role_ee['test_recall'].mean()
mean_f1_role_ee = cv_scores_role_ee['test_f1'].mean()
    # Fit on training data
ee_rf_role_classifier = ee_rf_role.fit(X_train, y2_train)
    # Prediction
ypred_test_role_ee = ee_rf_role_classifier.predict(X_test)
    # Metric on Testing data
test_accuracy_role_ee = accuracy_score(y2_test, ypred_test_role_ee)
test_precision_role_ee = precision_score(y2_test, ypred_test_role_ee, average='macro')
test_recall_role_ee = recall_score(y2_test, ypred_test_role_ee, average='macro')
test_f1_role_ee = f1_score(y2_test, ypred_test_role_ee, average='macro')
    # Classification Report to see break down of predictions
role_cr = classification_report(y2_test, ypred_test_role_ee, digits=4)


## Level
ee_rf_lvl = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
    # Cross-validation
cv_scores_lvl_ee = cross_validate(ee_rf_lvl, X_train, y3_train, cv=5, scoring=scoring)
    # Getting averaged metrics from cross-validation
mean_accuracy_lvl_ee = cv_scores_lvl_ee['test_accuracy'].mean()
mean_precision_lvl_ee = cv_scores_lvl_ee['test_precision'].mean()
mean_recall_lvl_ee = cv_scores_lvl_ee['test_recall'].mean()
mean_f1_lvl_ee = cv_scores_lvl_ee['test_f1'].mean()
    # Fit on training data
ee_rf_lvl_classifier = ee_rf_lvl.fit(X_train, y3_train)
    # Prediction
ypred_test_lvl_ee = ee_rf_lvl_classifier.predict(X_test)
    # Metric on Testing data
test_accuracy_lvl_ee = accuracy_score(y3_test, ypred_test_lvl_ee)
test_precision_lvl_ee = precision_score(y3_test, ypred_test_lvl_ee, average='macro')
test_recall_lvl_ee = recall_score(y3_test, ypred_test_lvl_ee, average='macro')
test_f1_lvl_ee = f1_score(y3_test, ypred_test_lvl_ee, average='macro')
    # Classification Report to see break down of predictions
lvl_cr = classification_report(y3_test, ypred_test_lvl_ee, digits=4)


def prediction_confidence(probabilities):
    return np.round(probabilities.max(axis=1)*100, 2)
# get lowest probability, map it to class
def anti_class(classifier, probabilities):
    classes = classifier.classes_
    min_prob = np.argmin(probabilities, axis=1)
    return [classes[i] for i in min_prob]

probabilities_func_ee = ee_rf_function_classifier.predict_proba(X_test)
probabilities_role_ee = ee_rf_role_classifier.predict_proba(X_test)
probabilities_lvl_ee = ee_rf_lvl_classifier.predict_proba(X_test)

pred_conf_func = prediction_confidence(probabilities_func_ee)
anti_class_func = anti_class(ee_rf_function_classifier, probabilities_func_ee)

pred_conf_role = prediction_confidence(probabilities_role_ee)
anti_class_role = anti_class(ee_rf_role_classifier, probabilities_role_ee)

pred_conf_lvl = prediction_confidence(probabilities_lvl_ee)
anti_class_lvl = anti_class(ee_rf_lvl_classifier, probabilities_lvl_ee)

predictions = {'Campaign Member ID (18)': ID_test, \
              'Title': X_test_og,\
              'Predicted Job Function': ypred_test_func_ee, \
              'Job Function Prediction Confidence': pred_conf_func, \
              'Anti-Function': anti_class_func, \
              'Predicted Job Role': ypred_test_role_ee, \
              'Job Role Prediction Confidence': pred_conf_role, \
              'Anti-Role': anti_class_role, \
              'Predicted Job Level': ypred_test_lvl_ee, \
              'Job Level Prediction Confidence': pred_conf_lvl, \
              'Anti-Level': anti_class_lvl}

predictions_df = pd.DataFrame(predictions)

# Get new Role prediction along with new metrics
# The model predicts Roles for Job Titles not in IT, this does not match the hierarchy because if the Function is not IT, 
# we want to return NON-ICP
predictions_df.loc[(predictions_df['Predicted Job Role'] != 'NON-ICP') & (predictions_df['Predicted Job Function'] != 'IT'), 'Predicted Job Role'] = 'NON-ICP'
role_pred_fixed = predictions_df['Predicted Job Role']
role_acc_fixed = accuracy_score(y2_test, role_pred_fixed)
role_precision_fixed = precision_score(y2_test, role_pred_fixed, average='macro')
role_recall_fixed = recall_score(y2_test, role_pred_fixed, average='macro')
role_f1_fixed = f1_score(y2_test, role_pred_fixed, average='macro')

# predictions_df.to_csv('predictions.csv', index=False)

# metrics
job_func_train = pd.DataFrame({'Job Function (Train)': [mean_accuracy_func_ee, mean_precision_func_ee, mean_recall_func_ee, mean_f1_func_ee]})
job_role_train = pd.DataFrame({'Job Role (Train)': [mean_accuracy_role_ee, mean_precision_role_ee, mean_recall_role_ee, mean_f1_role_ee]})
job_lvl_train = pd.DataFrame({'Job Level (Train)': [mean_accuracy_lvl_ee, mean_precision_lvl_ee, mean_recall_lvl_ee, mean_f1_lvl_ee]})
job_func_test = pd.DataFrame({'Job Function(Test)': [test_accuracy_func_ee, test_precision_func_ee, test_recall_func_ee, test_f1_func_ee]})
job_role_test = pd.DataFrame({'Job Role (Test)': [role_acc_fixed, role_precision_fixed, role_recall_fixed, role_f1_fixed]})
job_lvl_test = pd.DataFrame({'Job Level (Test)': [test_accuracy_lvl_ee, test_precision_lvl_ee, test_recall_lvl_ee, test_f1_lvl_ee]})
results = pd.concat([job_func_train, job_func_test, job_role_train, job_role_test, job_lvl_train, job_lvl_test], axis=1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
results.reset_index(drop=True, inplace=True)
results.index = metrics
