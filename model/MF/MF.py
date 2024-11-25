import numpy as np
import movielens
import pandas as pd
import splitter
from scipy.special import expit
    
from model.MF.preprocessing import ids_encoder


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
            self.history['epochs'].append(epoch)
            self.history['loss'].append(error)
            self.history['val_loss'].append(val_error)
            
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
    
    def evaluate(self, x_test, y_test):
        """
        compute the global error on the test set        
        :param x_test : test pairs (u,i) for which rating r_ui is known
        :param y_test : set of ratings r_ui for all test pairs (u,i)
        """
        error = self.mae(x_test, y_test)
        print(f"validation error : {round(error,3)}")
        
        return error
      
    def predict(self, userid, itemid):
        """
        Make rating prediction for a user on an item
        :param userid
        :param itemid
        :return r : predicted rating
        """
        # encode user and item ids to be able to access their latent factors in
        # matrices P and Q
        u = uencoder.transform([userid])[0]
        i = iencoder.transform([itemid])[0]

        # rating prediction using encoded ids. Dot product between P_u and Q_i
        r = np.dot(self.P[u], self.Q[i])
        return r

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
    

if __name__ == '__main__':
    df = movielens.load_pandas_df("1M")
    ratings = pd.DataFrame(df[["userID", "itemID", "rating"]])

    ratings, uencoder, iencoder = ids_encoder(ratings)

    m = ratings["userID"].nunique()   # total number of users
    n = ratings["userID"].nunique()   # total number of items
    split_df = splitter.interactive_split(ratings)

    x_train = split_df[0][["userID", "itemID"]].to_numpy()
    y_train = split_df[0][["rating"]].to_numpy().reshape(-1)

    x_test = split_df[1][["userID", "itemID"]].to_numpy()
    y_test = split_df[1][["rating"]].to_numpy().reshape(-1)


    MF = MatrixFactorization(m, n, k=10, alpha=0.01, lamb=1.5)

    history = MF.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    MF.evaluate(x_test, y_test)
