import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Ouvre l'experiment mLFlow pour logger le modèle et les métriques de référence
# Lancer MLFlow sur le port 5001 : mlflow ui --port 5001
# mlflow.set_tracking_uri("http://127.0.0.1:5001")

# EN PARAMETRE DES FONCTIONS : X et Y + best params : ATTENTION A DISSOCIER LES TABLES DU 1er ENTRAINEMENT DE CELLES DES REENTRAINMENTS


def search_hyperparam(X_train, y_train):
    # Recherche avec l'erreur OOB : method of measuring the prediction error of random forests
    n_estimators = [10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000]
    max_depth = [2, 5, 8, 10, 15, 20, 30, 40, 50, 100]
    best_oob_score = 0

    for nbrestim in n_estimators:
        for prof in max_depth:
            model = RandomForestClassifier(#class_weight='balanced', 
                                       bootstrap=True, 
                                       random_state=0, 
                                       oob_score=True)
            model.set_params(n_estimators=nbrestim, max_depth=prof)
            model.fit(X_train, y_train)
            oob_score = model.oob_score_ # donne l'oob_score sur l'échantillon out-of-bag (et donc sur le "jeu de validation")

            if oob_score > best_oob_score:
                best_oob_score = oob_score
                best_n_estimators = nbrestim
                best_max_depth = prof
            
    print("les meilleurs hyperparamètres sont : n_estimators =", best_n_estimators, "max_depth =", best_max_depth)
    print(f"Erreur OOB : {1 - model.oob_score_}")

    return best_n_estimators, best_max_depth


# FONCTION D'ENTRAINEMENT
def model_rfc(X_train, y_train, best_n_estim, best_max_depth, X_test, y_test):
    rfc = RandomForestClassifier(#class_weight='balanced',
                             n_estimators= best_n_estim,
                             max_depth= best_max_depth,
                                bootstrap=True, 
                                random_state=42, 
                                oob_score=True)

    rfc.fit(X_train, y_train)
    print(f"Erreur OOB : {1 - rfc.oob_score_}")

    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1, rfc
