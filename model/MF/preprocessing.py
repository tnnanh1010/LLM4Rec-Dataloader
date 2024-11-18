from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd


def mean_ratings(dataframe):
    means = dataframe.groupby(by='userID', as_index=False)['rating'].mean()
    return means


def normalized_ratings(dataframe, norm_column="norm_rating"):
    """
    Subscribe users mean ratings from each rating 
    """
    mean = mean_ratings(dataframe=dataframe)
    norm = pd.merge(dataframe, mean, suffixes=('', '_mean'), on='userid')
    norm[f'{norm_column}'] = norm['rating'] - norm['rating_mean']

    return norm

def ids_encoder(ratings, data_type="movielens", user_column="userID", item_column="itemID"):

    if item_column is None:
        item_column = "itemID" if data_type == 'movielens' else 'history'

    if data_type.lower() == 'movielens':
        users = sorted(ratings[user_column].unique())
        items = sorted(ratings[item_column].unique())

        # create users and items encoders
        uencoder = LabelEncoder()
        iencoder = LabelEncoder()

        # fit users and items ids to the corresponding encoder
        uencoder.fit(users)
        iencoder.fit(items)

        # encode userids and itemids
        ratings[user_column] = uencoder.transform(ratings[user_column].tolist())
        ratings[item_column] = iencoder.transform(ratings[item_column].tolist())
    elif data_type.lower() == 'mind':
        users = sorted(ratings[user_column].unique())
        items = sorted(set([item for sublist in ratings[item_column].apply(lambda x: x.split()).tolist() for item in sublist]))

        # create users and items encoders
        uencoder = LabelEncoder()
        iencoder = LabelEncoder()

        # fit users and items ids to the corresponding encoder
        uencoder.fit(users)
        iencoder.fit(items)

        # encode userids and itemids
        ratings[user_column] = uencoder.transform(ratings[user_column].tolist())

        # Now we need to handle history, positive, and negative as items list
        ratings[item_column] = ratings[item_column].apply(lambda x: [iencoder.transform([i])[0] for i in x.split()])


    return ratings, uencoder, iencoder

def format_data(train_df, dev_df, type='movielens'):

    if type.lower() =='movielens':
        x_train = train_df[["userID", "itemID"]].to_numpy()
        y_train = train_df[["rating"]].to_numpy().reshape(-1)

        x_test = dev_df[1][["userID", "itemID"]].to_numpy()
        y_test = dev_df[1][["rating"]].to_numpy().reshape(-1)
    
    elif type.lower() =='mind':
        x_train = train_df[["userID", "itemID"]].to_numpy()
        y_train = train_df[["rating"]].to_numpy().reshape(-1)

        x_test = dev_df[["userID", "itemID"]].to_numpy()
        y_test = dev_df[["rating"]].to_numpy().reshape(-1)


    return x_train, y_train, x_test, y_test

def preprocess_mind_data(behaviors_df):
    """
    Preprocess MIND dataset to extract the relevant information
    and format it for Matrix Factorization.
    """
    ratings = pd.DataFrame(columns=['userID', 'itemID', 'rating'])

    # Replace NaN in 'history' column with empty string
    behaviors_df['history'] = behaviors_df['history'].fillna('')

    # Create implicit feedback from history and positive items
    new_rows = []  # List to store new rows to be concatenated
    for _, row in behaviors_df.iterrows():
        user_id = row['userID']
        history = row['history']  # This will now be an empty string if original was NaN
        positive = row['positive'].split()
        negative = row['negative'].split()

        # For implicit feedback, consider positive items as 1 (interacted), negative as 0 (not interacted)
        for item in history.split():  # Safely split, even if history is an empty string
            new_rows.append({'userID': user_id, 'itemID': item, 'rating': 1})
        
        for item in positive:
            new_rows.append({'userID': user_id, 'itemID': item, 'rating': 1})
        
        for item in negative:
            new_rows.append({'userID': user_id, 'itemID': item, 'rating': 0})

    # Convert new rows into DataFrame and concatenate with the existing ratings DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    ratings = pd.concat([ratings, new_rows_df], ignore_index=True)

    # Encoding the user and item ids
    ratings, uencoder, iencoder = ids_encoder(ratings)

    return ratings, uencoder, iencoder