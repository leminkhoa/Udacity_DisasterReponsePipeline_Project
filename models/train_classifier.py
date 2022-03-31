# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import re
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from custom_transformer import StartingVerbExtractor

def load_data(database_filepath):
    # read in file
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_responses', con=engine)

    # define features and label arrays
    X = df['message']
    y = df.iloc[:, 4:]

    # Get category names
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    # text normalization
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]    
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # Add starting verb feature 
    pipeline = Pipeline([
                        ('features', FeatureUnion ([
                            ('text_pipeline', Pipeline ([
                                                ('vect', CountVectorizer(tokenizer=tokenize)),
                                                ('tfidf', TfidfTransformer())
                                            ])),      
                            ('starting_verb', StartingVerbExtractor ())   
                                    ])),        
                        ('clf', MultiOutputClassifier (xgb.XGBClassifier(eval_metric='logloss')))
                        ])

    # Set parameters
    parameters = {
              'clf__estimator__min_child_weight' : [1, 5],
              'clf__estimator__gamma' : [0, 1],
              'clf__estimator__max_depth': [10, 6],
              'clf__estimator__max_delta_step': [0, 1],
              'clf__estimator__n_estimators' : [150],
              'clf__estimator__eta' : [0.3, 0.2],
              'clf__estimator__use_label_encoder': [False]
            }

    # Create gridsearch for pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=4, cv=3)
    return model

def metrics_evaluation(y_true, y_pred):
    test_results = []
    report_cols = ['column', 'precision_0', 'precision_1', 'recall_0', 'recall_1',
                   'f1_score_0', 'f1_score_1', 'accuracy']
    # Loop through
    for index, col in zip(range(y_true.shape[1]), y_true.columns):
        report_dict = classification_report(y_true.iloc[:,index].values, 
                                            y_pred[:, index],
                                            output_dict=True,
                                            zero_division=0)
        report_df = pd.DataFrame.from_dict(report_dict)
        # Get precision
        try:
            precision_0 = report_df.loc['precision', '0']
        except Exception:
            precision_0 = np.NaN
        try:
            precision_1 = report_df.loc['precision', '1']
        except Exception: 
            precision_1 = np.NaN 
        # Get recall
        try:
            recall_0 = report_df.loc['recall', '0']
        except Exception:
            recall_0 = np.NaN
        try:
            recall_1 = report_df.loc['recall', '1']
        except Exception: 
            recall_1 = np.NaN 
        # Get f1 score
        try:
            f1_score_0 = report_df.loc['f1-score', '0']
        except Exception:
            f1_score_0 = np.NaN
        try:
            f1_score_1 = report_df.loc['f1-score', '1']
        except Exception: 
            f1_score_1 = np.NaN
        # Get accuracy
        accuracy = report_df['accuracy'][0]
        # Collect results
        test_results.append([col, precision_0, precision_1, recall_0, recall_1, f1_score_0, f1_score_1, accuracy])
    # Create df after the loop
    report_df = pd.DataFrame(data=test_results, columns=report_cols)
    # Get mean of metrics
    metrics_tup = namedtuple('Metrics', ['precision_0', 'precision_1', 'recall_0',
                                         'recall_1', 'f1_0', 'f1_1', 'acc'])
    mean_precision_0 = report_df['precision_0'].mean()
    mean_precision_1 = report_df['precision_1'].mean()
    mean_recall_0 = report_df['recall_0'].mean()
    mean_recall_1 = report_df['recall_1'].mean()
    mean_f1_0 = report_df['f1_score_0'].mean()
    mean_f1_1 = report_df['f1_score_1'].mean()
    mean_acc = report_df['accuracy'].mean()
    metrics_mean = metrics_tup(mean_precision_0, mean_precision_1, mean_recall_0, mean_recall_1, 
                               mean_f1_0, mean_f1_1, mean_acc)
    return report_df, metrics_mean

def evaluate_model(model, X_test, y_test, category_names):
    print("    Model best params: ", model.best_params_)

    # Predict
    y_pred = model.predict(X_test)

    # Get returned metrics report
    report_df, metrics_mean = metrics_evaluation(y_test, y_pred)

    # Print result for each category
    print('CATEGORY RESULTS:')
    for category_name in category_names:
        category_acc = report_df.loc[report_df['column']==category_name, 'accuracy'].values[0]
        precision_0 = report_df.loc[report_df['column']==category_name, 'precision_0'].values[0]
        precision_1 = report_df.loc[report_df['column']==category_name, 'precision_1'].values[0]
        recall_0 = report_df.loc[report_df['column']==category_name, 'recall_0'].values[0]
        recall_1 = report_df.loc[report_df['column']==category_name, 'recall_1'].values[0]
        f1_score_0 = report_df.loc[report_df['column']==category_name, 'f1_score_0'].values[0]
        f1_score_1 = report_df.loc[report_df['column']==category_name, 'f1_score_1'].values[0]
        print(f'- Category name: {category_name!r}')
        print(f'    - Accuracy : {category_acc:.2%}')
        print(f'    - Precision of class 0 : {precision_0:.2%}')
        print(f'    - Precision of class 1 : {precision_1:.2%}')
        print(f'    - Recall of class 0 : {recall_0:.2%}')
        print(f'    - Recall of class 1 : {recall_1:.2%}')
        print(f'    - F1-score of class 0 : {f1_score_0:.2%}')
        print(f'    - F1-score of class 1 : {f1_score_1:.2%}')
    # Print average evaluating results for whole model
    print('OVERALL MODEL RESULTS:')
    print('- Accuracy: {:.2%}'.format(metrics_mean[6]))
    print('- Precision of class 0: {:.2%}'.format(metrics_mean[0]))
    print('- Precision of class 1: {:.2%}'.format(metrics_mean[1]))
    print('- Recall of class 0: {:.2%}'.format(metrics_mean[2]))
    print('- Recall of class 1: {:.2%}'.format(metrics_mean[3]))
    print('- F1 of value 0: {:.2%}'.format(metrics_mean[4]))
    print('- F1 of value 1: {:.2%}'.format(metrics_mean[5]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()