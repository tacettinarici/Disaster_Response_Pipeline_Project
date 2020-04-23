import sys

# Data Munging & Cleaning libraries
import pandas as pd
import numpy as np
import re

# Import Machine Learning libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import Natural Lang. Proc. libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Library to pickle the model
#from joblib import dump
from sklearn.externals import joblib


# Set connection to database with sqlite3 library
import sqlite3

nltk.download(['punkt', 'wordnet', 'stopwords'])

'''
Example of execution:
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
'''

def load_data(database_filepath):
    '''
    Data is loaded and 'X' / 'Y'  values are defined. After that dropped unused features and duplicates.
    Args:
        database_filepath  : (relative) filepath of sqlite database
    Returns:
        X and Y as feature and target dataset respectively without duplicates
    '''
    # Commence connection to database
    conn = sqlite3.connect(database_filepath)
    # Read data from SQL, the tablename is ''Disaster_ETL'' which is created through the ETL_pipeline.
    df = pd.read_sql('SELECT * FROM Disaster_ETL', conn)
    # Values needs to be 1-d for tf-idf transformations.
    X = df.message.tolist()
    # After checking value counts of columns, looks like for this dataset it makes sense to remove 'child_alone'.
    y = df.drop(columns=['child_alone', 'message', 'original','id', 'genre'], axis=1)
    # Get all category names that we are trying to predict
    target_names = y.columns
    return X, y, target_names


def tokenize(text):
    '''
    Tokenization of the text file
    Args:
        text : Message column which needs NLP preprocessing
    Returns:
        Cleaned from punctation, case sensitivity,stopwords. Also  all words are lemmatized and stemmed.
    '''
    # Cleaned from punctation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower()
    
    # Word tokenize text
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Words are lemmatized and stemmed
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if not word in stop_words]
    return tokens


def build_model():
    '''
    Return a pipeline that applies 'vect', 'tf-idf' transformation and 'Random Forest Classifier' to the data
    Args:
        None
    Returns:
        A pipeline object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_leaf=1)))
    ])
    return pipeline





def evaluate_model(model, X_test, Y_test, target_names):
    '''
    Prints the performance of the model for all the different categories that we are trying to predict.
    Args:
        model          : Makes the predictions 
        X_test         : Features to be used for predictions
        Y_test         : Target Value
        target_names   : Name of Target Columns
    Returns:
        None
    '''
    # Predict the X_test values
    Y_pred = model.predict(X_test)
    # Display of each Predictons
    idx = -1
    for column in target_names:
        idx += 1
        print(column)
        print('_'*60)
        print(classification_report(Y_test[column], Y_pred[:,idx]))
        print("Accuracy: {0:.4f}".format(accuracy_score(Y_test[column], Y_pred[:,idx])))
        print("\n")
    
    
    

def save_model(model, model_filepath):
    '''
    Trained model is saved as pickle file
    Args:
        model          : Makes the predictions 
        model_filepath : path where to keep the pickled model
    Returns:
        None
    '''
    joblib.dump(model, model_filepath,compress=3)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, target_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, target_names)

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