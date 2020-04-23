import sys
# import libraries for loading data
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Process:
        1.Load the data from two csv files
        2.Combined dataset after removed from duplicate rows
    
    Args:
        messages_filepath  : (relative) filepath of messages.csv
        categories_filepath: (relative) filepath of categories.csv

    Returns:        
        Returned to 'df'
    '''
    # loading raw datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
 
    # Dropping duplicates rows from the datasets
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
   
    # Remove double id columns in categories
    categories.drop_duplicates(subset='id', keep="last", inplace=True)

    # Merge the datasets
    df = messages.merge(categories, how='outer',on='id')
    
    return df
    


def clean_data(df):
    '''
    Process:
        Clean the category column from 'df' and make columns for all values

    Args:
        df: Pandas dataframe

    Returns:
        Returned to Cleaned category dataset that call 'df'
    '''    
        
    # Preparing  'categories' from dataframe
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # Change the column names
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
        categories[column] = categories[column].astype(int)
    
    # Merging new 'categories' to dataframe
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat((df, categories), axis=1)
    
    # Dropping duplicates rows from the 'df' dataset
    df.drop_duplicates(subset='id',inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Process:
        Store the dataframe in a sqlite database on your local machine

    Args:
        df               : Pandas dataframe
        database_filename: (relative) filepath of sqlite database

    Returns:
        None
    '''
    conn = sqlite3.connect(database_filename)
    df.to_sql('Disaster_ETL', conn, index=False, if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()