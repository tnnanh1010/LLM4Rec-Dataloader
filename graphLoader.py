import random
import numpy as np
import pandas as pd
import scipy.sparse as sp



class ImplicitCF(object):
    """Data processing class for GCN models which use implicit feedback.

    Initialize train and test set, create normalized adjacency matrix and sample data for training epochs.

    """

    def __init__(
        self,
        train,
        test,
        col_user='userID',
        col_item='itemID',
        col_rating='rating',
        seed=None,
    ):
        """
        Args:
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            seed (int): Seed.

        """
        self.user_idx = None
        self.item_idx = None
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.train, self.test = self.reindex_user_item(train, test)
        self.train_users = self.train[self.col_user].nunique()
        self._init_train_data()

        random.seed(seed)


    def reindex_user_item(self, train, test):
        """
        Reindex the user_id and item_id columns and return the DataFrame with new indices.

        Args:
            df (pandas.DataFrame): The input dataframe containing user-item interactions.
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).

        Returns:
            pandas.DataFrame: Test and Train DataFrame with reindexed user and item columns.
        """
        
        df = pd.concat([train, test], axis=0, ignore_index=True)
        self.n_users = df[self.col_user].nunique()
        self.n_items = df[self.col_item].nunique()

        # Step 1: Reindex users and items by creating integer indices
        user_idx = pd.Series(df[self.col_user].unique()).reset_index(drop=True).reset_index()
        user_idx.columns = ["user_idx", self.col_user]
        
        item_idx = pd.Series(df[self.col_item].unique()).reset_index(drop=True).reset_index()
        item_idx.columns = ["item_idx", self.col_item]
        
        # Step 2: Merge the new indices back into the original DataFrame
        train = pd.merge(train, user_idx, on=self.col_user, how="left")
        train = pd.merge(train, item_idx, on=self.col_item, how="left")
        
        test = pd.merge(test, user_idx, on=self.col_user, how="left")
        test = pd.merge(test, item_idx, on=self.col_item, how="left")

        # Ensure the new indices are within the range
        train[self.col_user] = train["user_idx"]
        train[self.col_item] = train["item_idx"]
        test[self.col_user] = test["user_idx"]
        test[self.col_item] = test["item_idx"]

        train = train.drop(columns=["user_idx", "item_idx", "timestamp"])
        test = test.drop(columns=["user_idx", "item_idx", 'timestamp'])

        return train, test


    def _init_train_data(self):
        """Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.

        """
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[self.train[self.col_user], self.train[self.col_item]] = 1.0
        # print(self.R)

    def get_norm_adj_mat(self):
        """Load normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        norm_adj_mat = self.create_norm_adj_mat()
        
        return norm_adj_mat

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("Already create adjacency matrix.")

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
        print("Already normalize adjacency matrix.")

        return norm_adj_mat.tocsr()

    def train_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """

        def sample_neg(interacted_items):
            """Sample a negative item for a user, ensuring it is not already interacted with."""
            if len(interacted_items) >= self.n_items:
                raise ValueError("A user has voted in every item. Can't find a negative sample.")
            neg_id = random.choice([item for item in range(self.n_items) if item not in interacted_items])
            return neg_id
            # Randomly select users
        users = random.sample(range(self.train_users), batch_size)

        # Get interactions for sampled users
        interact = self.interact_status.iloc[users]
        
        # Sample positive and negative items
        pos_items = interact[self.col_item + "_interacted"].apply(lambda x: random.choice(list(x)))
        neg_items = interact[self.col_item + "_interacted"].apply(sample_neg)


        return np.array(users), np.array(pos_items), np.array(neg_items)