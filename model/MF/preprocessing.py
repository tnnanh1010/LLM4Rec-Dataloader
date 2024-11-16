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

def ids_encoder(ratings, user_column="userID", item_column="itemID"):
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

    return ratings, uencoder, iencoder

