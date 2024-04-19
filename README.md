
# Netskope Practicum (2024 Team 3)

The goal of this practicum project is to create a solution that is able to predict job function, role, and level using the job title and puts emphasis on accuracy over runtime to ensure the correct customer is targeted.

## Zip File Structure
In the zipped file we include our data cleaning Jupyter Notebook that walks through the process of cleaning up the source of truth (Historical Lead Records.csv). Outputted from the Data Cleaning.ipynb file is the cleaned data set, cleaned_data.csv. 

We also include python files for the models we built which include NLTK's Naive Bayes, Complement Naive Bayes, random forest, Gradient Boosting, AdaBoost, and XGBoost. These files contain our code for cross-validation, training, testing, and evaluation. Random Forest was able to obtain the highest accuracy and a good f1-score for classes in the ICP. Thus, random_forest.py outputs the prediction of the testing set from the SoT.

The top performing model, random forest, was put into the top_performance.py which includes functions to clean the incoming data and preprocess our given source of truth for training, lines of code to train the random forest model and predict F/R/L, and lastly code to monitor the model performance using accuracy, precision, recall, and f1-score. The output is a dataframe containing unique campaign ID, Job Title, predicted Job Function, Anti-Function (the function it is least likely to be), predicted Job Role, Anti-Role, Job Level, Anti-Level, and the prediction confidence percentages of each prediction.  
```bash
├── Historical Lead Records.csv
├── Data Cleaning.ipynb
│   ├── cleaned_data.csv
│       ├── complement_naive_bayes.py
│       ├── nltk_naive_bayes.py
│       ├── XBG.py
│       ├── gradient_boosting.py
│       ├── adaboost.py
│       ├── random_forest.py
|         └── predictions.csv
└── top_performance.py
    
```
## Installation and Setup
In order to get predictions, only top_performance.py needs to be run. The first line of code is a placeholder for a file path that should be replaced with the ingested data in the form of a .csv file. The clean_data function will clean the data and the rest of the script will train the model, gather predictions along with anti-predictions and the prediction confidence percentage. Outputted will be a predictions.csv containing the aforementioned columns. 
### Python Libraries Used
We used python version 3.11.5 for this project.
##### For cleaning the data:
- pandas
- numpy
- re
- string

##### For preprocessing the cleaned data:
- spacy
- nltk
- stopwords from nltk.corpus
- text from sklearn.feature_extraction
- TfidfVectorizer from sklearn.feature_extraction.text
- train_test_split from sklearn.model_selection

##### For modeling:
- EasyEnsembleClassifier from imblearn.ensemble
- RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier from from sklearn.ensemble
- XGBClassifier from xgboost
- ComplementNB from sklearn.naive_bayes
- NaiveBayesClassifier from nltk.classify
- GridSearchCV from sklearn.model_selection


##### For evaluation:
- accuracy_score, precision_score, f1_score, recall_score, classification_report from sklearn.metrics
- cross_validate from sklearn.model_selection

## Data
We use various regex functions to rid the data of junk job titles that include phone numbers, emails, titles with special characters or punctuations, and null values. We then employ the logic of replacing R/F/L associated with duplicate Job Titles with the most frequent observation. The data is also double checked for C-Level/Director/Executive correctness to ensure less errors in prediction. 

### Data Preprocessing
The cleaned data is stratified split with 70% for training and 30% for testing. The stratified splitting ensures that the proportion of classes in training and testing is the same as it is in the original, full dataset. The job titles are then removed of stopwords, excluding words of the form "it", "its", "it's" to keep assure information security is still present in the data. After training and testing, Job titles are vectorized using TF-IDF to prevent data leakage.

## Results and Evaluation
Random forest performs the best with the highest accuracies across all classes and a balanced f1-score. The classification report for random forest further proves this statement. We can distinguish a notable decrease in the Job Role when applying the logic to match the predicted values to the hierarchy. However, overall we see accuracies of 90% across the classes and a relative balanced f1-score. For metrics related to the ICP, we find that random forest does extremely well at correctly predicting positive values. 

|                | Job Function (Train) | Job Function (Test) | Job Role (Train) | Job Role (Test) | Job Level (Train) | Job Level (Test) |
| -------------  | -------------------- | -------------------| ------------------- | ------------------- | ------------------- | ------------------- |
|  **Accuracy**  |  0.8964              | 0.9065              | 0.9000  | 0.8881 | 0.9048 | 0.9100 |
|  **Precision** |  0.6085              | 0.6335              | 0.7380  |  0.7666 | 0.7842 | 0.7884 |
|  **Recall**    |  0.9362              | 0.9456              | 0.9037  |  0.7864 | 0.8824 | 0.8931 |
|  **F1-Score**  |  0.6891              | 0.7175              | 0.7912  | 0.7724 | 0.7797 | 0.7844 |

| | Precision | Recall | F1-score |
| ------ | ------ | ------ | ------|
| **Function=IT**| 0.9888 | 0.9081 | 0.9468 |
|**Level=C-Level** | 0.9254 | 0.9339 | 0.9296 |
|**Level=Director** | 0.9426 | 0.9654 | 0.9539 |
|**Level=Executive** | 0.9058 | 0.9148 | 0.9102 |

## Authors
- [@Kelly Lam](https://www.linkedin.com/in/koding-with-kelly/)
- Raja Kolli
- Anhtuan Ho


