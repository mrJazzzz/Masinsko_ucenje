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
# |
# coordinate descent: regresija (X,y,lambda) -> model

# evaluation(X_test,y_test) -> metrike (rmse)

# split (X,y,k,i) -> X_train,y_train,X_test,y_test

# cross_validation_kfold (X,y,k) -> metrike (rmse - usrednjeno po foldovima)



# za k-fold jednom samo da permutujem mesta i onda redom uzimam k-tine





