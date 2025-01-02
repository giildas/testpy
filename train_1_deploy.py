import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlflow.models.signature import infer_signature

import requests

import time

# Ouvre l'experiment mLFlow pour logger le modèle et les métriques de référence
# Lancer MLFlow sur le port 5001 : mlflow ui --port 5001
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# EN PARAMETRE DES FONCTIONS : X et Y + best params : ATTENTION A DISSOCIER LES TABLES DU 1er ENTRAINEMENT DE CELLES DES REENTRAINMENTS

# CHARGEMENT DES DATASETS : depuis la commande de la page yml ??
# ATTENTION : 2 TABLES MONITORING DIFFERENTES
# Appel de l'API pour récupérer les données
url = "http://localhost:8001/data/contxt"  # Remplacez par l'URL de votre API
response = requests.get(url)

# Test générique : Vérifier qu'il ne manque pas de labellisation de segments
data = response.json()
df_contxt = pd.DataFrame(data)

embedding_df = pd.DataFrame(df_contxt['embedding'].to_list(), columns=[f'emb_{i}' for i in range(len(df_contxt['embedding'][0]))])
print(embedding_df.head(1))

# Fusionner les nouvelles colonnes avec le DataFrame original
df = pd.concat([df_contxt.drop(columns=['embedding']), embedding_df], axis=1)
array= df.to_numpy()


X = array[:, 5:] # vecteurs d'embedding
print("X_1 :", X[1,:])
print("len X_1 :", len(X[1,:]))
Y = array[:, 3]
print("Y_1.unique() :", np.unique(Y))
X = X.astype(float)
Y = Y.astype(int)
print(X.dtype)
print(Y.dtype)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

# Inclure un test sur la distribution des données ?

# A injecter dans MLFlow
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5), dpi=100)

plt.title("Distribution des données")

plt.hist([Y, y_train, y_test],
         bins = [x - 0.5 for x in range(0, 3)],
         rwidth=0.75,
         label=["Données originales", "Données d'apprentissage", "Données de test"])

plt.annotate("n = " + str(np.sum(Y == 0)), (-0.25, 365.0), ha='center')
plt.annotate("n = " + str(np.sum(y_train == 0)), (0.0, 293.0), ha='center')
plt.annotate("n = " + str(np.sum(y_test == 0)), (0.25, 80.0), ha='center')

plt.annotate("n = " + str(np.sum(Y == 1)), (0.75, 220.0), ha='center')
plt.annotate("n = " + str(np.sum(y_train == 1)), (1.0, 178.0), ha='center')
plt.annotate("n = " + str(np.sum(y_test == 1)), (1.25, 50.0), ha='center')

plt.xticks(np.arange(2), ("non_Offre", "Offre"))

plt.xlabel("Label")
plt.ylabel("Nombre d'exemples")

plt.ylim(0.0, 2000.0)

plt.legend()

plt.show()

# RECHERCHE DES MEILLEURS HYPERPARAMETRES
# Recherche avec l'erreur OOB : method of measuring the prediction error of random forests
start = time.time()

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

end = time.time()
temps_hyperparam = (end - start) / 60
print('temps passé pour la recherche des hyperparamètres : ', temps_hyperparam) 


# ENTRAINEMENT AVEC MEILLEURS HYPERPARAMETRES
rfc = RandomForestClassifier(#class_weight='balanced',
                             n_estimators= best_n_estimators,
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
print("accuracy :", accuracy)
print("precision :", precision)
print("recall :", recall)
print("f1 :", f1)


# LOGS MLFLOW
'''with mlflow.start_run(run_name="model_metrics_train") as run:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    # Inférer la signature une fois le modèle entraîné
    X_train_df = pd.DataFrame(X_train)
    input_example = X_train_df.sample(1)  # Exemple d'entrée (une seule ligne)
    signature = infer_signature(X_train_df, rfc.predict(X_train))  # Signature

    # Logger les métriques
    mlflow.log_metric("accuracy_rfc_1", accuracy)
    mlflow.log_metric("precision_rfc_1", precision)
    mlflow.log_metric("recall_rfc_1", recall)
    mlflow.log_metric("f1_score_rfc_1", f1)

    # Logger les hyperparamètres
    mlflow.log_param("n_estimators_rfc_1", best_n_estimators)
    mlflow.log_param("max_depth_rfc_1", best_max_depth)

    # Logger le modèle avec signature et exemple d'entrée
    # 
    mlflow.sklearn.log_model(
        rfc,
        "rfc_1",
        input_example=input_example,
        signature=signature
    )'''