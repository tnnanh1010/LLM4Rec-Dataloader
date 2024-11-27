import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix


class MatrixFactorization:
    
    def __init__(self, m, n, k=10, alpha=0.001, lamb=0.01):
        """
        Initialization of the model        
        : param
            - m : number of users
            - n : number of items
            - k : length of latent factor, both for users and items. 
            - alpha : learning rate. 0.001 by default
            - lamb : regularizer parameter. 0.02 by default
        """
        np.random.seed(32)
        
        # initialize the latent factor matrices P and Q (of shapes (m,k) and (n,k) respectively) that will be learnt
        self.k = k
        self.P = np.random.normal(size=(m, k))
        self.Q = np.random.normal(size=(n, k))


        # hyperparameter initialization
        self.alpha = alpha
        self.lamb = lamb
        
        # training history
        self.history = {
            "epochs":[],
            "loss":[],
            "val_loss":[],
            "lr":[]
        }
    
    def print_training_parameters(self):
        print('Training Matrix Factorization Model ...')
        print(f'k={self.k} \t alpha={self.alpha} \t lambda={self.lamb}')
    
    def update_rule(self, u, i, error):
        self.P[u] = self.P[u] + self.alpha * (error * self.Q[i] - self.lamb * self.P[u])
        self.Q[i] = self.Q[i] + self.alpha * (error * self.P[u] - self.lamb * self.Q[i])
        
    def mae(self,  x_train, y_train):
        """
        returns the Mean Absolute Error
        """
        # number of training exemples
        M = x_train.shape[0]
        error = 0
        for pair, r in zip(x_train, y_train):
            u, i = pair
            error += abs(r - np.dot(self.P[u], self.Q[i]))
        return error/M

    def bce_loss(self, x, y):
        """
        Compute Binary Cross-Entropy loss for given inputs
        """
        loss = 0
        for pair, r in zip(x, y):
            u, i = pair
            r_hat = expit(np.dot(self.P[u], self.Q[i]))  # Sigmoid activation
            # Compute BCE loss
            loss += -(r * np.log(r_hat + 1e-9) + (1 - r) * np.log(1 - r_hat + 1e-9))
        return loss / len(y)
    
    def print_training_progress(self, epoch, epochs, error, val_error, steps=5):
        if epoch == 1 or epoch % steps == 0 :
                print("epoch {}/{} - loss : {} - val_loss : {}".format(epoch, epochs, round(error,3), round(val_error,3)))
                
    def learning_rate_schedule(self, epoch, target_epochs = 20):
        if (epoch >= target_epochs) and (epoch % target_epochs == 0):
                factor = epoch // target_epochs
                self.alpha = self.alpha * (1 / (factor * 20))
                print("\nLearning Rate : {}\n".format(self.alpha))
    
    def fit(self, x_train, y_train, validation_data, epochs=1000, loss='mae'):
        """
        Train latent factors P and Q according to the training set
        
        :param
            - x_train : training pairs (u,i) for which rating r_ui is known
            - y_train : set of ratings r_ui for all training pairs (u,i)
            - validation_data : tuple (x_test, y_test)
            - epochs : number of time to loop over the entire training set. 
            1000 epochs by default
            
        Note that u and i are encoded values of userid and itemid
        """
        self.print_training_parameters()
        
        # validation data
        x_test, y_test = validation_data
        
        # loop over the number of epochs
        for epoch in range(1, epochs+1):
            
            # for each pair (u,i) and the corresponding rating r
            for pair, r in zip(x_train, y_train):
                
                # get encoded values of userid and itemid from pair
                u,i = pair
                
                # compute the predicted rating r_hat
                r_hat = np.dot(self.P[u], self.Q[i])
                
                # compute the prediction error
                e = abs(r - r_hat)
                
                # update rules
                self.update_rule(u, i, e)
                
            # training and validation error  after this epochs
            if loss == 'mae':
                error = self.mae(x_train, y_train)
                val_error = self.mae(x_test, y_test)
            elif loss == 'bce':
                error = self.bce_loss(x_train, y_train)
                val_error = self.bce_loss(x_test, y_test)
            
            # update history
            self.update_history(epoch, error, val_error)
            
            # print training progress after each steps epochs
            self.print_training_progress(epoch, epochs, error, val_error, steps=1)
              
            # leaning rate scheduler : redure the learning rate as we go deeper in the number of epochs
            # self.learning_rate_schedule(epoch)
        
        return self.history
    
    def update_history(self, epoch, error, val_error):
        self.history['epochs'].append(epoch)
        self.history['loss'].append(error)
        self.history['val_loss'].append(val_error)
        self.history['lr'].append(self.alpha)
    
    def evaluate(self, x_test, y_test, loss='mae'):
        """
        compute the global error on the test set        
        :param x_test : test pairs (u,i) for which rating r_ui is known
        :param y_test : set of ratings r_ui for all test pairs (u,i)
        """
        if loss == 'mae':
            error = self.mae(x_test, y_test)
        elif loss == 'bce':
            error = self.bce_loss(x_test, y_test)
        print(f"validation error : {round(error,3)}")
        
        return error
      
    def recommend(self, userid, N=30):
        """
        make to N recommendations for a given user

        :return(top_items,preds) : top N items with the highest predictions 
        with their corresponding predictions
        """
        # encode the userid
        u = uencoder.transform([userid])[0]

        # predictions for users userid on all product
        predictions = np.dot(self.P[u], self.Q.T)

        # get the indices of the top N predictions
        top_idx = np.flip(np.argsort(predictions))[:N]

        # decode indices to get their corresponding itemids
        top_items = iencoder.inverse_transform(top_idx)

        # take corresponding predictions for top N indices
        preds = predictions[top_idx]

        return top_items, preds     


class MatrixFactorizationBPR:
    def __init__(self, m, n, k=10, alpha=0.01, lamb=0.001):
        """
        Initialize Matrix Factorization with BPR loss.
        :param
            - m : number of users
            - n : number of items
            - k : length of latent factors
            - alpha : learning rate
            - lamb : regularization parameter
        """
        np.random.seed(32)
        self.n_user = m
        self.n_item = n
        self.k = k
        self.P = np.ones(shape=(m, k))
        self.Q = np.ones(shape=(n, k)) 
        self.alpha = alpha
        self.lamb = lamb
        self.history = {"epochs": [], "loss": [], "lr": [], "train_auc": [], "val_auc": []}
    
    def print_training_parameters(self):
        print(f"Training Matrix Factorization with BPR Loss...")
        print(f"k={self.k}, alpha={self.alpha}, lambda={self.lamb}")
    
    def bpr_loss(self, u, i, j):
        """
        Compute the BPR loss on the training data.
        """

        x_uij = np.dot(self.P[u], self.Q[i] - self.Q[j])
 
        sigmoid = 1 / (1 + np.exp(-x_uij))

        #L2 normalization
        regularization = self.lamb / 2 * (np.linalg.norm(self.P[u])**2 + 
                                          np.linalg.norm(self.Q[i])**2 + 
                                          np.linalg.norm(self.Q[j])**2)

        loss = -np.log(sigmoid) + regularization
        return loss
    
    def update_rule(self, u, i, j):
        """
        Perform SGD update for a single triplet (u, i, j).
        """
        x_uij = np.dot(self.P[u], self.Q[i] - self.Q[j])


        sigmoid_gradient = np.exp(x_uij)  

        
        # Update user and item latent factors
        self.P[u] += self.alpha * ((sigmoid_gradient / (1 + sigmoid_gradient))  * (self.Q[i] - self.Q[j]) - self.lamb * self.P[u])
        self.Q[i] += self.alpha * ((sigmoid_gradient / (1 + sigmoid_gradient)) * self.P[u] - self.lamb * self.Q[i])
        self.Q[j] += self.alpha * (-(sigmoid_gradient / (1 + sigmoid_gradient)) * self.P[u] - self.lamb * self.Q[j])
        
    def auc_score(self, bpr_matrix):
        """
        Compute the AUC score based on a pandas DataFrame for BPR matrix.
        
        :param bpr_matrix: pandas DataFrame with columns ['userID', 'itemID', 'rating']
        :return: AUC score
        """
        auc = 0.0
        
        for u in range(self.n_user):
            
            y_pred = self.P[u] @ self.Q.T
            # y_pred = a[u]

            # Initialize the true labels as a zero array
            y_true = np.zeros(self.n_item)

            y_true[bpr_matrix[u].indices] = 1

            if len(np.unique(y_true)) == 1:
                continue

            # Calculate the AUC for this user and add it to the total AUC
            auc += roc_auc_score(y_true, y_pred)
            
        
        return auc / self.n_user

    def convert_to_bpr_mat(self, dataframe, threshold=3):

        tempdf = dataframe
        tempdf['positive'] = tempdf['rating'].apply(func=lambda x: 0 if x < threshold else 1)

        # Vì tập dữ liệu này đánh index từ 1 nên chuyển sang kiểu category
        # để tránh việc chúng ta có ma trận
        tempdf['userID'] = tempdf['userID'].astype('category')
        tempdf['itemID'] = tempdf['itemID'].astype('category')

        bpr_mat = csr_matrix((tempdf['positive'],
                            (tempdf['userID'].cat.codes,
                            tempdf['itemID'].cat.codes)))
        bpr_mat.eliminate_zeros()
        del tempdf
        return bpr_mat

    def update_history(self, epoch, error):
        self.history['epochs'].append(epoch)
        self.history['loss'].append(error)
        self.history['lr'].append(self.alpha)


    def fit(self, train_df, test_df, epochs=100):
        """
        Train the model using BPR loss and evaluate on validation data.
        :param
            - x_train : training pairs (u, i)
            - y_train : training ratings
            - x_test : validation pairs (u, i)
            - n_items : total number of items in the dataset
        """
        self.print_training_parameters()
        
         # Convert the input DataFrame into BPR format
        bpr_train_mat = self.convert_to_bpr_mat(train_df)
        bpr_test_mat = self.convert_to_bpr_mat(test_df)


        pos = np.split(bpr_train_mat.indices, bpr_train_mat.indptr)[1:-1]
        neg = [np.setdiff1d(np.arange(0, self.n_item,1), e) for e in pos]


        for epoch in range(1, epochs + 1):
            # Training
            u = np.random.randint(0, self.n_user)
            i = pos[u][np.random.randint(0, len(pos[u]))]
            j = neg[u][np.random.randint(0, len(neg[u]))]
            self.update_rule(u, i, j)
            
            # Compute training loss (BPR)
            train_loss = self.bpr_loss(u, i, j)
            
            # Update history
            self.update_history(epoch, train_loss)
            
            print(f"Epoch {epoch}/{epochs} - BPR Loss: {train_loss:.4f} ")

        train_auc = self.auc_score(bpr_train_mat)
        test_auc = self.auc_score(bpr_test_mat)

        print(f"Train AUC: {train_auc:.4f} - Val AUC: {test_auc:.4f}")

        return self.history
    
    def predict(self):
        return self.P @ self.Q.T

'''
if __name__ == "__main__":

    df = movielens.load_pandas_df("100K")
    
    ratings = pd.DataFrame(df[["userID", "itemID", "rating"]])

    ratings, uencoder, iencoder = ids_encoder(ratings)

    m = ratings["userID"].nunique()  # number of users
    n = ratings["itemID"].nunique()  # number of items

    split_df = splitter.interactive_split(ratings)

    MF_BPR = MatrixFactorizationBPR(m, n, k=12, alpha=0.01, lamb=0.01)

    MF_BPR.fit(split_df[0], split_df[1], epochs=10000)
'''
