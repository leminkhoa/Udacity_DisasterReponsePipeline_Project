import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load raw data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Combine dataframes
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    # Create a dataframe of individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda string: string[:-2]) # Exclude the last 2 characters, ex: related[-1], request[-0]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 to 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # Replace 'categories' column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    try:
        df.to_sql('disaster_responses', engine, index=False)
    except ValueError as err:
        print("Table 'disaster_responses' already exists")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {messages}\n    CATEGORIES: {categories}'
              .format(messages=messages_filepath, categories=categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {database}'.format(database=database_filepath))
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