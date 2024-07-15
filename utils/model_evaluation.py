from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_dev': scores.std()
    }
