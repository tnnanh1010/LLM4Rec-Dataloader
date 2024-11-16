import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from sklearn.metrics import mean_squared_error
import numpy as np

class MatrixFactorization:
    def __init__(self, factors=50, regularization=0.1, iterations=20, alpha=1.0):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model = AlternatingLeastSquares(factors=self.factors,
                                             regularization=self.regularization,
                                             iterations=self.iterations)
        self.user_mapping = {}
        self.item_mapping = {}
    
    def load_data(self, filepath, dataset_type='movielens'):
        if dataset_type == 'movielens':
            df = pd.read_csv(filepath)
            user_category = df['userID'].astype("category")
            item_category = df['itemID'].astype("category")

            # Encode userID and itemID
            df['user_id_encoded'] = user_category.cat.codes
            df['item_id_encoded'] = item_category.cat.codes

            # Create mappings
            self.user_mapping = dict(enumerate(user_category.cat.categories))
            self.item_mapping = dict(enumerate(item_category.cat.categories))

            # Create user-item matrix
            self.user_item_matrix = csr_matrix((df['rating'], (df['user_id_encoded'], df['item_id_encoded'])))
        else:
            raise ValueError("Unsupported dataset type. Choose either 'movielens' or 'mind'.")
        return df

    
    def train_test_split(self, df, test_size=0.2):
        train, test = train_test_split(df, test_size=test_size, random_state=42)

        # Use the full set of categories for encoding
        train_matrix = csr_matrix(
            (train['rating'] if 'rating' in train else np.ones(len(train)),
            (train['user_id_encoded'], train['item_id_encoded']))
        )
        test_matrix = csr_matrix(
            (test['rating'] if 'rating' in test else np.ones(len(test)),
            (test['user_id_encoded'], test['item_id_encoded']))
        )

        # Check for unmapped `item_id_encoded` in the test set
        unmapped_items = set(test['item_id_encoded']).difference(self.item_mapping.keys())
        if unmapped_items:
            raise ValueError(f"Test set contains unmapped items: {unmapped_items}")

        return train_matrix, test_matrix

    def fit(self, train_matrix):
        """
        Fit the ALS model to the training data.
        """
        self.model.fit(train_matrix.T * self.alpha)
    
    def recommend_for_user(self, user_id, num_recommendations=10):
        """
        Generate recommendations for a given user.
        """
        if user_id not in self.user_mapping:
            print(f"User ID {user_id} not found.")
            return []
        
        user_idx = list(self.user_mapping.keys())[list(self.user_mapping.values()).index(user_id)]
        recommendations = self.model.recommend(user_idx, self.user_item_matrix[user_idx], N=num_recommendations)
        
        
        # print(recommendations)
        # print(recommendations.info())
        # print(self.item_mapping.keys())
        # return [(self.item_mapping[item_id], score) for item_id, score in zip(recommendations[0], recommendations[1])]
        return [(self.item_mapping.get(item_id, "Unknown Item"), score) for item_id, score in zip(recommendations[0], recommendations[1])]

    def evaluate_mse(self, test_matrix):
        mse = 0
        count = 0
        
        # Ensure test_matrix is in COO format for row/col access
        test_coo = test_matrix.tocoo()
        
        # Get the number of users and items from the model's factor matrices
        num_users, num_items = self.model.user_factors.shape
        
        # Iterate through the test matrix
        for user, item, actual_rating in zip(test_coo.row, test_coo.col, test_coo.data):
            # Ensure user and item are within bounds
            if user >= num_users or item >= num_items:
                print(f"Warning: Skipping user {user} or item {item} - out of bounds")
                continue
            
            # Get the predicted rating
            prediction = self.model.user_factors[user] @ self.model.item_factors[item].T
            
            # Calculate MSE
            mse += (prediction - actual_rating) ** 2
            count += 1
        
        mse /= count
        return mse



if __name__ == '__main__':
    # Instantiate the MF class
    mf = MatrixFactorization()

    # Load MovieLens dataset
    movielens_data = mf.load_data('output.csv', dataset_type="movielens")

    # Split into training and testing sets
    train_matrix, test_matrix = mf.train_test_split(movielens_data)
    

    # Train the model
    mf.fit(train_matrix)

    # Get recommendations for a user
    recommendations = mf.recommend_for_user(user_id=1)
    print("Recommendations for user 1:", recommendations)

    # Evaluate model on test set
    mse = mf.evaluate_mse(test_matrix)
    print("Mean Squared Error on test set:", mse)

