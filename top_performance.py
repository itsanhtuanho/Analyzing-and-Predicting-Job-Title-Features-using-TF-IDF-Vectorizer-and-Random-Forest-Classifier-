
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from imblearn.ensemble import EasyEnsembleClassifier


# path of data csv
path = 'data.csv'
data = pd.read_csv(path)

# Function to clean data
def clean_data(df):
    # drop null values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    #replaces special character with nothing
    def remove_special_charac(title):
        return re.sub('[^\x00-\x7F]', '', title)
    df['Title'] = df['Title'].apply(remove_special_charac)

    # removes emails
    def is_email(text):
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.match(email_regex, text) is not None
    
    # removes titles with all numbers
    def is_all_numbers(text):
        return text.isdigit()
    
    # removes titles that is literally just a phone number
    def is_phone_num(text):
        phone_regex = r'\b(?:\d{1}-)?\(?(\d{3})\)?[-. ]?(\d{3})[-. ]?(\d{4})\b'
        return re.match(phone_regex, text) is not None
    
    # removes titles that only have punctuation
    def has_only_punctuation(text):
        return bool(re.fullmatch(r'[^A-Za-z0-9\s]+', text))
    
    # filters data using 4 previous functions
    remove_mask = df['Title'].apply(is_email) | df['Title'].apply(is_all_numbers) | df['Title'].apply(is_phone_num) | df['Title'].apply(has_only_punctuation)
    df = df[~remove_mask]

    df = df[df['Title'].str.len() > 1]
    df = df[df['Title'] != 'NA']

    # removes punctuation in title
    df['Title'] = df['Title'].apply(lambda x: re.sub('[{}]'.format(string.punctuation), '', x))
    
    # removes any digits from title
    def remove_digits(title):
        digits = r'\b\S*\d\S*\b'
        return re.sub(digits, '', title)
    df['Title'] = df['Title'].apply(remove_digits)

    # remove trailing and leading whitespaces
    df['Title'] = df['Title'].str.strip()
    # convert everything in df to uppercase, except for 'Campaign Member ID (18)'AssertionError column
    df.loc[:, df.columns != 'Campaign Member ID (18)'] = df.loc[:, df.columns != 'Campaign Member ID (18)'].applymap(lambda x: x.upper() if isinstance(x, str) else x)
    #rename to cleaned_df
    cleaned_df = df.copy()
    return cleaned_df


cleaned_df = clean_data(data)
cleaned_df_titles = cleaned_df['Title']
cleaned_df_ID = cleaned_df['Campaign Member ID (18)']
## Uncomment the following code to get accuracy of newly made test set 
# y1_test = cleaned_df['Job Function']
# y2_test = cleaned_df['Job Role']
# y3_test = cleaned_df['Job Level']


data_for_training = pd.read_csv('cleaned_data.csv', keep_default_na=False)
#Function for additional preprocessing (train-test split, stopword removal, tfidf)
def preprocess_train(train_data, test_data):
    # isolate target variables
    y_combined = train_data[['Job Function', 'Job Role', 'Job Level']]
    # train test split 70/30
    X_train_og, X_test_og, y_combined_train, y_combined_test, ID_train, ID_test = train_test_split(train_data['Title'], y_combined, train_data['Campaign Member ID (18)'], test_size=0.3, random_state=69, stratify=y_combined)
    # split y variables for TRAINING and TESTING
    y1_train, y2_train, y3_train = y_combined_train.iloc[:, 0], y_combined_train.iloc[:, 1], y_combined_train.iloc[:, 2]
    
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
    X_test = tfidf_vectorizer.transform(test_data['Title'])
    return X_train, X_test, ID_test, y1_train, y2_train, y3_train


X_train, X_test, ID_test, y1_train, y2_train, y3_train = preprocess_train(data_for_training, cleaned_df)

# Modeling (Using our own cleaned_data.csv as a SoT for training)
rf = RandomForestClassifier(n_estimators=100, random_state=69)

## Function (remove verbose=2 if user does not want to keep track of model training progress)
ee_rf_func = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
ee_rf_function_classifier = ee_rf_func.fit(X_train, y1_train)
ypred_test_func_ee = ee_rf_function_classifier.predict(X_test)

## Role 
ee_rf_role = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
ee_rf_role_classifier = ee_rf_role.fit(X_train, y2_train)
ypred_test_role_ee = ee_rf_role_classifier.predict(X_test)

## Level 
ee_rf_lvl = EasyEnsembleClassifier(base_estimator=rf, random_state=69, verbose=2)
ee_rf_lvl_classifier = ee_rf_lvl.fit(X_train, y3_train)
ypred_test_lvl_ee = ee_rf_lvl_classifier.predict(X_test)


## Prediction
def prediction_confidence(probabilities):
    return np.round(probabilities.max(axis=1)*100, 2)
# get lowest probability, map it to class
def anti_class(classifier, probabilities):
    classes = classifier.classes_
    min_prob = np.argmin(probabilities, axis=1)
    return [classes[i] for i in min_prob]

probabilities_func_ee = ee_rf_function_classifier.predict_proba(cleaned_df_titles)
probabilities_role_ee = ee_rf_role_classifier.predict_proba(cleaned_df_titles)
probabilities_lvl_ee = ee_rf_lvl_classifier.predict_proba(cleaned_df_titles)

pred_conf_func = prediction_confidence(probabilities_func_ee)
anti_class_func = anti_class(ee_rf_function_classifier, probabilities_func_ee)

pred_conf_role = prediction_confidence(probabilities_role_ee)
anti_class_role = anti_class(ee_rf_role_classifier, probabilities_role_ee)

pred_conf_lvl = prediction_confidence(probabilities_lvl_ee)
anti_class_lvl = anti_class(ee_rf_lvl_classifier, probabilities_lvl_ee)

predictions = {'Campaign Member ID (18)': cleaned_df_ID, \
              'Title': cleaned_df_titles,\
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
# Get new Role predictions
# The model predicts Roles for Job Titles not in IT, this does not match the hierarchy because if the Function is not IT, 
# we want to return NON-ICP
predictions_df.loc[(predictions_df['Predicted Job Role'] != 'NON-ICP') & (predictions_df['Predicted Job Function'] != 'IT'), 'Predicted Job Role'] = 'NON-ICP'
role_pred_fixed = predictions_df['Predicted Job Role']
# predictions df to csv file
predictions_df.to_csv('predictions.csv', index=False)


## Metrics to validate performance
def model_performance_metrics(y1_test, y2_test, y3_test, function_pred, role_pred, lvl_pred):
        # Func
    test_accuracy_func = accuracy_score(y1_test, function_pred)
    test_precision_func = precision_score(y1_test, function_pred, average='macro')
    test_recall_func = recall_score(y1_test, function_pred, average='macro')
    test_f1_func = f1_score(y1_test, function_pred, average='macro')
        # Role
    test_accuracy_role = accuracy_score(y2_test, role_pred)
    test_precision_role = precision_score(y2_test, role_pred, average='macro')
    test_recall_role = recall_score(y2_test, role_pred, average='macro')
    test_f1_role = f1_score(y2_test, role_pred, average='macro')
        # Level
    test_accuracy_lvl = accuracy_score(y3_test, lvl_pred)
    test_precision_lvl = precision_score(y3_test, lvl_pred, average='macro')
    test_recall_lvl = recall_score(y3_test, lvl_pred, average='macro')
    test_f1_lvl = f1_score(y3_test, lvl_pred, average='macro')

    job_func_test = pd.DataFrame({'Job Function (Test)': [test_accuracy_func, test_precision_func, test_recall_func, test_f1_func]})
    job_role_test = pd.DataFrame({'Job Role (Test)': [test_accuracy_role, test_precision_role, test_recall_role, test_f1_role]})
    job_lvl_test = pd.DataFrame({'Job Level (Test)': [test_accuracy_lvl, test_precision_lvl, test_recall_lvl, test_f1_lvl]})
    results = pd.concat([job_func_test, job_role_test, job_lvl_test], axis=1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    results.reset_index(drop=True, inplace=True)
    results.index = metrics
    return results

## Uncomment the following code to get accuracy of newly made test set 
# results = model_performance_metrics(y1_test, y2_test, y3_test, ypred_test_func_ee, role_pred_fixed, ypred_test_lvl_ee)



