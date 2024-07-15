from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)
