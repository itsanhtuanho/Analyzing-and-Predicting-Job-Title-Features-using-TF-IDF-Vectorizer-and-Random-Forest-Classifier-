# %%
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

# read in cleaned data
data_for_training = pd.read_csv('cleaned_data.csv', keep_default_na=False)

# remove stopwords
combined_stop_words = set(stopwords.words('english')).union(set(text.ENGLISH_STOP_WORDS))

if 'it' in combined_stop_words:
    combined_stop_words.remove('it')

if 'its' in combined_stop_words:
    combined_stop_words.remove('its')

if "it's" in combined_stop_words:
    combined_stop_words.remove("it's")

# function to format title for classifier
def get_title_features(title):
    features = {}
    word_tokens = nltk.word_tokenize(title)
    filtered_words = [w for w in word_tokens if not w.lower() in combined_stop_words] 
    for word in filtered_words:
        features['contains({})'.format(word.lower())] = True
    if len(filtered_words) > 0:
        first_key = 'first({})'.format(filtered_words[0].lower())
        last_key = 'last({})'.format(filtered_words[-1].lower())
        features[first_key] = True
        features[last_key] = True
    return features

# initialize an empty list to store the formatted data
raw_job_titles = []

# iterate over the rows of the DataFrame
for index, row in data_for_training.iterrows():
    # create a dictionary for the current row
    job_data = {
        "Title": row['Title'],
        "Job Function": row['Job Function'],
        "Job Role": row['Job Role'],
        "Job Level": row['Job Level']
    }
    # append the dictionary to the list
    raw_job_titles.append(job_data)

#function
function_features = [
    (
         get_title_features(job_title["Title"]),
         job_title["Job Function"]
    )
    for job_title in raw_job_titles
    if job_title["Job Function"] is not None
]

#role
role_features = [
    (
         get_title_features(job_title["Title"]),
         job_title["Job Role"]
    )
    for job_title in raw_job_titles
    if job_title["Job Role"] is not None
]

#level
level_features = [
    (
         get_title_features(job_title["Title"]),
         job_title["Job Level"]
    )
    for job_title in raw_job_titles
    if job_title["Job Level"] is not None
]

# Splitting data for job function
function_features_train, function_features_test = train_test_split(function_features, test_size=0.3, random_state=69)

# Splitting data for job role
role_features_train, role_features_test = train_test_split(role_features, test_size=0.3, random_state=69)

# Splitting data for job level
level_features_train, level_features_test = train_test_split(level_features, test_size=0.3, random_state=69)

# create a function to train and evaluate naivebayesclassifier
def train_and_test_classifier(features_train, features_test, target_name):
    # train the classifier
    classifier = NaiveBayesClassifier.train(features_train)

    # initialize dictionaries to store metrics for both training and test data
    metrics = {}

    # evaluate performance on training data
    train_refsets = {label: set() for (feats, label) in features_train}
    train_testsets = {label: set() for label in classifier.labels()}

    for i, (feats, label) in enumerate(features_train):
        train_refsets[label].add(i)
        observed = classifier.classify(feats)
        train_testsets[observed].add(i)

    # evaluate performance on test data
    test_refsets = {label: set() for (feats, label) in features_test}
    test_testsets = {label: set() for label in classifier.labels()}

    for i, (feats, label) in enumerate(features_test):
        test_refsets[label].add(i)
        observed = classifier.classify(feats)
        test_testsets[observed].add(i)

    # calculate and store metrics for both training and test data for each label
    for label in classifier.labels():
        train_precision = nltk.precision(train_refsets[label], train_testsets[label])
        train_recall = nltk.recall(train_refsets[label], train_testsets[label])
        train_f1 = nltk.f_measure(train_refsets[label], train_testsets[label])

        test_precision = nltk.precision(test_refsets[label], test_testsets[label])
        test_recall = nltk.recall(test_refsets[label], test_testsets[label])
        test_f1 = nltk.f_measure(test_refsets[label], test_testsets[label])

        # calculate support
        train_support = len(train_refsets[label])
        test_support = len(test_refsets[label])

        # calculate accuracy for training and test data
        train_accuracy = nltk.classify.accuracy(classifier, features_train)
        test_accuracy = nltk.classify.accuracy(classifier, features_test)

        metrics[f"{label} (Train)"] = [train_accuracy, train_precision, train_recall, train_f1, train_support]
        metrics[f"{label} (Test)"] = [test_accuracy, test_precision, test_recall, test_f1, test_support]

    # compute and store average metrics
    avg_train_precision = sum(metrics[f"{label} (Train)"][1] for label in classifier.labels()) / len(classifier.labels())
    avg_train_recall = sum(metrics[f"{label} (Train)"][2] for label in classifier.labels()) / len(classifier.labels())
    avg_train_f1 = sum(metrics[f"{label} (Train)"][3] for label in classifier.labels()) / len(classifier.labels())

    avg_test_precision = sum(metrics[f"{label} (Test)"][1] for label in classifier.labels()) / len(classifier.labels())
    avg_test_recall = sum(metrics[f"{label} (Test)"][2] for label in classifier.labels()) / len(classifier.labels())
    avg_test_f1 = sum(metrics[f"{label} (Test)"][3] for label in classifier.labels()) / len(classifier.labels())

    metrics[f"{target_name} (Train)"] = [train_accuracy, avg_train_precision, avg_train_recall, avg_train_f1, len(features_train)]
    metrics[f"{target_name} (Test)"] = [test_accuracy, avg_test_precision, avg_test_recall, avg_test_f1, len(features_test)]

    # calculate confusion matrix for the test set
    test_true_labels = [label for (_, label) in features_test]
    test_predicted_labels = [classifier.classify(feats) for (feats, _) in features_test]
    confusion_matrix = nltk.ConfusionMatrix(test_true_labels, test_predicted_labels)

    return metrics, confusion_matrix, test_predicted_labels

# train and evaluate job function
function_metrics, function_confusion, function_predictions = train_and_test_classifier(function_features_train, function_features_test, "Job Function")
print(function_metrics)
print(function_confusion)

# convert function classfication report to dataframe
function_class_report = pd.DataFrame.from_dict(function_metrics, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
function_class_report = function_class_report.iloc[1::2]

function_index_mapping = {
    'IT (Test)': 'IT',
    'ENGINEERING (Test)': 'ENGINEERING',
    'NON-ICP (Test)': 'NON-ICP',
    ' (Test)': ' ',
    'RISK/LEGAL/COMPLIANCE (Test)': 'RISK/LEGAL/COMPLIANCE',
    'PROCUREMENT (Test)': 'PROCUREMENT',
    'Job Function (Test)': 'Overall'
}
function_class_report = function_class_report.rename(index=function_index_mapping)
function_class_report.loc[function_class_report.index != 'Overall', 'Accuracy'] = ''
print(function_class_report)

#train and evaluate job role
role_metrics, role_confusion, role_predictions = train_and_test_classifier(role_features_train, role_features_test, "Job Role")
print(role_metrics)
print(role_confusion)

# convert role classfication report to dataframe
role_class_report = pd.DataFrame.from_dict(role_metrics, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
role_class_report = role_class_report.iloc[1::2]

role_index_mapping = {
    'INFORMATION SECURITY (Test)': 'INFORMATION SECURITY',
    'NON-ICP (Test)': 'NON-ICP',
    'IT GENERAL (Test)': 'IT GENERAL',
    'NETWORKING (Test)': 'NETWORKING',
    'SYSTEMS (Test)': 'SYSTEMS',
    'DEVELOPMENT (Test)': 'DEVELOPMENT',
    'Job Role (Test)': 'Overall'
}
role_class_report = role_class_report.rename(index=role_index_mapping)
role_class_report.loc[role_class_report.index != 'Overall', 'Accuracy'] = ''
print(role_class_report)

#train and evaluate job level
level_metrics, level_confusion, level_predictions = train_and_test_classifier(level_features_train, level_features_test, "Job Level")
print(level_metrics)
print(level_confusion)

# convert level classfication report to dataframe
level_class_report = pd.DataFrame.from_dict(level_metrics, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
level_class_report = level_class_report.iloc[1::2]

level_index_mapping = {
    'CONTRIBUTOR (Test)': 'CONTRIBUTOR',
    'MANAGER (Test)': 'MANAGER',
    'C-LEVEL (Test)	': 'C-LEVEL',
    'DIRECTOR (Test)': 'DIRECTOR',
    'EXECUTIVE (Test)': 'EXECUTIVE',
    ' (Test)': ' ',
    'Job Level (Test)': 'Overall'
}
level_class_report = level_class_report.rename(index=level_index_mapping)
level_class_report.loc[level_class_report.index != 'Overall', 'Accuracy'] = ''
print(level_class_report)

# metrics for all target variables, train and test
function_metrics_dict = dict(list(function_metrics.items())[-2:])
role_metrics_dict = dict(list(role_metrics.items())[-2:])
level_metrics_dict = dict(list(level_metrics.items())[-2:])

combined_dict = {}
combined_dict.update(function_metrics_dict)
combined_dict.update(role_metrics_dict)
combined_dict.update(level_metrics_dict)

combined_dict = {k: v[:4] for (k, v) in combined_dict.items()}

nb_metrics_df = pd.DataFrame(data=combined_dict)

custom_index = ["Accuracy", "Precision", "Recall", "F1-Score"]

nb_metrics_df.index = custom_index

print(nb_metrics_df)

# create predictions df to change predictions
predictions = {
              'Predicted Job Function': function_predictions, \
              'Predicted Job Role': role_predictions, \
              'Predicted Job Level': level_predictions, \
              }

predictions_df = pd.DataFrame(predictions)     

#extract true role labels from test set
role_features_test_list = [tuple[1] for tuple in role_features_test]

#updated role metrics for roles that are under 'IT' but function =! 'IT'
predictions_df.loc[(predictions_df['Predicted Job Role'] != 'NON-ICP') & (predictions_df['Predicted Job Function'] != 'IT'), 'Predicted Job Role'] = 'NON-ICP'

role_pred_fixed = predictions_df['Predicted Job Role']
role_acc_fixed = accuracy_score(role_features_test_list, role_pred_fixed)
role_precision_fixed = precision_score(role_features_test_list, role_pred_fixed, average='macro')
role_recall_fixed = recall_score(role_features_test_list, role_pred_fixed, average='macro')
role_f1_fixed = f1_score(role_features_test_list, role_pred_fixed, average='macro')

print("adjusted accuracy on test set for job role is:", role_acc_fixed)
print("adjusted precision on test set for job role is:", role_precision_fixed)
print("adjusted recall on test set for job role is:", role_recall_fixed)
print("adjusted f1 score on test set for job role is:", role_f1_fixed)


