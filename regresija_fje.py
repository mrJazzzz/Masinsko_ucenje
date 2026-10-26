import numpy as np
from itertools import combinations_with_replacement

# Generisanje obelezja za polinomijalnu regresiju
def generate_polynomial_features(X: np.ndarray, p: int) -> np.ndarray:
    """
    Generise polinomna obelezja na osnovu matrice obelezja X do stepena p .

    :param X: Ulazna matrica obelezja dimenzija (n,m) - n je broj primera a m broj obelezja
    :param p: Mamsimalni stepen polinoma,
    :return: Matrica sa svim obelezjima X_out.
    """
    n, m = X.shape
    cols = []

    # Generisem sve kombinacije sa ponavljanjem
    for d in range(1, p + 1):
        for combo in combinations_with_replacement(range(m), d):
            col = np.prod(X[:, combo], axis=1, keepdims=True)
            cols.append(col)

    return np.hstack(cols) if cols else np.empty((n, 0))

# korak za coordinate descent (X,y,i, lambda)
# coordinate_descent_step(X:np.ndarray, y:np.ndarray, i:int, lambda:float)
# |
# coordinate_descent_regression(X:np.ndarray,y:np.ndarray,lambda:float) -> model
#   initialize model (parameters)
#   i = 0
#   while (convergence/iteration condition):
#       coordinate_descent_step(X,y,i,lambda)
#       i+=1; i%=dim

# evaluation(X_test: np.ndarray,y_test:np.ndarray, model) -> metrike (rmse):
#   calculate rmse manually

# split_fold (X:np.ndarray,ynp.ndarray,k:int,i:int) -> X_train,y_train,X_test,y_test:
#   k for k-fold, i is the number of fold

# The other option is to provide the number of examples and then return indexes and have a bit
# different cross_validation_kfold function


# cross_validation_kfold (X,y,k) -> metrike (rmse - usrednjeno po foldovima)
#   one permutation for data and then using split_fold function to take data
# using split function, training and evaluating model, averaged


# plots from the text





