import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(data, ratio=0.75, seed=42):
    """Pandas random splitter.

    The splitter randomly splits the input data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio
            of training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.

    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    multi_split, ratio = process_split_ratio(ratio)

    if multi_split:
        splits = split_pandas_data_with_ratios(data, ratio, shuffle=True, seed=seed)
        splits_new = [x.drop("split_index", axis=1) for x in splits]

        return splits_new
    else:
        return train_test_split(data, test_size=None, train_size=ratio, random_state=seed)

def chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user='userID',
    col_item='itemID',
    col_timestamp='timestamp',
):
    # A few preliminary checks.
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    if col_user not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")

    if col_item not in data.columns:
        raise ValueError("Schema of data not valid. Missing Item Col")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    order_by = col_timestamp

    data = data.sort_values([split_by_column, order_by])

    groups = data.groupby(split_by_column)

    data["count"] = groups[split_by_column].transform("count")
    data["rank"] = groups.cumcount() + 1

    

    splits = []
    prev_threshold = None
    for threshold in np.cumsum(ratio):
        condition = data["rank"] <= round(threshold * data["count"])
        if prev_threshold is not None:
            condition &= data["rank"] > round(prev_threshold * data["count"])
        splits.append(data[condition].drop(["rank", "count"], axis=1))
        prev_threshold = threshold

    return splits

import pandas as pd
import numpy as np

def interactive_split(data, ratio=[0.8, 0.2], seed=42):
    """
    Splits the dataset into train, validation, and test sets for each user.
    This function ensures all sets contain data for all users and uses random splitting.

    Args:
        data (pd.DataFrame): Input dataset with at least 'userID' and 'itemID'.
        ratio (float or list): Split ratio for the data. Can be a single float (e.g., 0.75 for train-test split)
                               or a list of floats summing to 1 (e.g., [0.6, 0.2, 0.2] for train-val-test split).
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of DataFrames (e.g., [train, test] or [train, val, test]).
    """
    np.random.seed(seed)

    # Process the ratio
    multi_split, ratio = process_split_ratio(ratio)

    print(ratio)

    # Initialize lists for splits
    splits = [[] for _ in range(len(ratio))]

    for user, group in data.groupby("userID"):
        n_items = len(group)
        indices = np.arange(n_items)
        np.random.shuffle(indices)

        # Calculate cumulative split indices
        split_indices = np.cumsum([round(r * n_items) for r in ratio[:-1]])
        split_slices = np.split(indices, split_indices)

        # Assign splits
        for i, split_slice in enumerate(split_slices):
            splits[i].append(group.iloc[split_slice])

    # Combine all splits
    combined_splits = [pd.concat(split).reset_index(drop=True) for split in splits]
    return combined_splits


def sequential_split(data, ratio=[0.8, 0.2]):
    """
    Splits the dataset into train, validation, and test sets for each user chronologically.
    Ensures the chronological order is preserved within each split.

    Args:
        data (pd.DataFrame): Input dataset with at least 'userID', 'itemID', and 'timestamp'.
        ratio (float or list): Split ratio for the data. Can be a single float (e.g., 0.75 for train-test split)
                               or a list of floats summing to 1 (e.g., [0.6, 0.2, 0.2] for train-val-test split).

    Returns:
        list: List of DataFrames (e.g., [train, test] or [train, val, test]).
    """
    # Process the ratio
    multi_split, ratio = process_split_ratio(ratio)

    # Initialize lists for splits
    splits = [[] for _ in range(len(ratio))]

    for user, group in data.groupby("userID"):
        group = group.sort_values("timestamp")
        n_items = len(group)

        # Calculate cumulative split indices
        split_indices = np.cumsum([round(r * n_items) for r in ratio[:-1]])
        split_slices = np.split(np.arange(n_items), split_indices)

        # Assign splits
        for i, split_slice in enumerate(split_slices):
            splits[i].append(group.iloc[split_slice])

    # Combine all splits
    combined_splits = [pd.concat(split).reset_index(drop=True) for split in splits]
    return combined_splits

def process_split_ratio(ratio):
    """Generate split ratio lists.

    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).

    Returns:
        tuple:
        - bool: A boolean variable multi that indicates if the splitting is multi or single.
        - list: A list of normalized split ratios.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")

        multi = False
    elif isinstance(ratio, list):
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if math.fsum(ratio) != 1.0:
            ratio = [x / math.fsum(ratio) for x in ratio]

        multi = True
    else:
        raise TypeError("Split ratio should be either float or a list of floats.")

    return multi, ratio

def min_rating_filter(
    data,
    min_rating=1,
    filter_by="user",
    col_user="userID",
    col_item='itemID',
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pandas.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = col_user if filter_by == "user" else col_item


    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return data.groupby(split_by_column).filter(lambda x: len(x) >= min_rating)

def split_pandas_data_with_ratios(data, ratios, seed=42, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if math.fsum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits

if __name__ == "__main__":
    print(1)