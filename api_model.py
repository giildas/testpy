# Lancer FastAPI avec uvicorn : uvicorn api_model:app --host 127.0.0.1 --port 8000
# Lancer MLFlow sur le port 5001 : mlflow ui --port 5001 / mlflow server --host 127.0.0.1 --port 5001
# Lancer l'appli app.py sur le port 5000

from fastapi import FastAPI, Request, Query, HTTPException
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from joblib import load
import httpx

from collections import Counter

import mlflow
import mlflow.sklearn

from functions import lignes_segm, best_skill, check_performance_degradation, send_alert, delete_training_data_1, delete_training_data_2
from train import search_hyperparam, model_rfc

from sklearn.model_selection import train_test_split

# Initialiser l'application FastAPI
app = FastAPI()

# CHARGEMENT DU MODELE DE LANGUE
model = SentenceTransformer("Lajavaness/sentence-camembert-large")

# CHARGEMENT DES MODELES DEPUIS MLFLOW
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Suivi évolution des modèles PCO")
#rfc_1 = mlflow.sklearn.load_model("models:/rfc_1/Production")
#rfc_2 = mlflow.sklearn.load_model("models:/rfc_2/Production")


rfc_1 = load('C:\\Users\\Utilisateur\\Documents\\Prepa_Diplome\\PCO_dec\\Algos\\rfc_1.joblib')
rfc_2 = load('C:\\Users\\Utilisateur\\Documents\\Prepa_Diplome\\PCO_dec\\Algos\\rfc_2.joblib')


@app.post("/predict")
def index(request: Request, text_input: str = Query(None, description="Texte à analyser")):
    segmented_text = None
    results_algo_1 = []
    results_algo_2 = []
    results_comp = []

    if request.method == "POST" and text_input:
        # PHASE 1 : SEGMENTATION
        segmented_text = lignes_segm(text_input)

        # PHASE 2 : ALGO_1 : prédictions avec rfc_1
        for segment in segmented_text:
            segm_embed = model.encode([segment], show_progress_bar=True)

            offre_predict = rfc_1.predict(segm_embed)
            print("offre_predict :", offre_predict)

            results_algo_1.append({
                    'segment': segment,
                    'segm_embed': segm_embed.flatten().tolist(),
                    'offre_predict': int(offre_predict[0])
                })

        df_results_algo_1 = pd.DataFrame(results_algo_1)
        y_pred_algo_1 = df_results_algo_1['offre_predict']

        # PHASE 3 : ALGO_2 : prédictions avec rfc_2 sur les segments classés comme [1]
        df_offre = df_results_algo_1.loc[df_results_algo_1['offre_predict'] == 1]

        for _, row in df_offre.iterrows():
            vecteur = row['segm_embed']
            segment = row['segment']
            comp_predict = rfc_2.predict([vecteur])
            print("comp_predict :", comp_predict)

            results_algo_2.append({
                'segment': segment,
                'segm_embed': vecteur,
                'comp_predict': int(comp_predict[0])
            })

        df_competences = pd.DataFrame(results_algo_2)
        y_pred_algo_2 = df_competences['comp_predict'].tolist()

        # PHASE 4 : FONCTION SIMILARITE
        df_final = df_competences.loc[df_competences['comp_predict'] == 1]
        print('df_final :', df_final)

        for _, row in df_final.iterrows():
            segment = row['segment']
            segm_embed = row['segm_embed']

            comp_dict = best_skill(segm_embed)
            print("comp_dict:", comp_dict)
            if comp_dict:
                comp_ref = list(comp_dict.keys())[0]
            else:
                comp_ref = "Aucune compétence trouvée pour le segment"
            print("comp_ref:", comp_ref)

            results_comp.append({
                'segment': segment,
                'competence_referentiel': comp_dict
            })
        
            print("results_comp après ajout :", results_comp)

        # Convertir  `results_algo_1', `results_algo_2` et `results_comp` en format JSON-serialisable
        results_algo_1_serializable = [{
            "segment": result["segment"],
            "offre_predict": result["offre_predict"]
        } for result in results_algo_1]
        
        results_algo_2_serializable = [{
            "segment": result["segment"],
            "comp_predict": result["comp_predict"]
        } for result in results_algo_2]

        for result in results_comp:
            print(result)

        results_comp_serializable = [{
            "segment": result["segment"],
            "competence_referentiel": result["competence_referentiel"]
        } for result in results_comp]

        print("results_algo_1 :", results_algo_1)
        print("results_algo_2 :", results_algo_2)
        print("results_comp :", results_comp)

    # Retourner les données JSON
    return {
        "nombre_segments_algo_1" : sum(y_pred_algo_1),
        "nombre_segments_algo_2": sum(y_pred_algo_2),
        "segmented_text": segmented_text,
        "results_algo_1": results_algo_1_serializable,
        "results_algo_2": results_algo_2_serializable,
        "results_comp": results_comp_serializable
    }


# ROUTE RETRAIN : COMMENCE PAR RECUPERER LES DONNEES DES 2 TABLES DE MONITORING
@app.post("/train_algo_1")
async def train():
    async with httpx.AsyncClient() as client:
        # Récupérer les données de la table /data/contxt
        try:
            response_contxt = await client.get("http://127.0.0.1:8001/data/contxt") 
            response_contxt.raise_for_status()
            data_contxt = response_contxt.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données de /data/contxt: {e}")
    df_contxt = pd.DataFrame(data_contxt)
    print("nb lignes dans df_contxt :", len(df_contxt))

    # PREPARER LES DATASETS : ATTENTION : prendre que les feedback=='agree'
    df_contxt = df_contxt.loc[df_contxt['feedback_user'] == 'agree']

    array_contxt = df_contxt.to_numpy()

    X_algo_1 = np.array(df_contxt['embedding'].to_list())

    array_contxt = np.concatenate([array_contxt, X_algo_1], axis=1)
    print("arr_contxt : ", array_contxt.shape)

        # PREPA DATA ALGO 1
    Y_algo_1 = array_contxt[:, 3]
    X_algo_1 = X_algo_1.astype(float)
    Y_algo_1 = Y_algo_1.astype(int)
    
    X_train_algo_1, X_test_algo_1, y_train_algo_1, y_test_algo_1 = train_test_split(X_algo_1, Y_algo_1, test_size=0.20, random_state=42, stratify=Y_algo_1)

    
    # IMPORTER les fonctions de RECHERCHE D'HYPERPARAMETRES PUIS D'ENTRAINEMENT
    n_estimators_algo_1, max_depth_algo_1 = search_hyperparam(X_train_algo_1, y_train_algo_1)

    accuracy_new_algo_1, precision_new_algo_1, recall_new_algo_1, f1_new_algo_1, new_rfc_1 = model_rfc(X_train_algo_1, y_train_algo_1, 
                                                                                                       n_estimators_algo_1, max_depth_algo_1, 
                                                                                                       X_test_algo_1, y_test_algo_1)

    # TESTER LES PERFORMANCES DES NOUVEAUX MODELES:
    with mlflow.start_run(run_name='Suivi Algo_1') as run_algo_1:
    # Log des métriques pour rfc_1
        mlflow.log_metric("accuracy_algo_1", accuracy_new_algo_1)
        mlflow.log_metric("precision_algo_1", precision_new_algo_1)
        mlflow.log_metric("recall_algo_1", recall_new_algo_1)
        mlflow.log_metric("f1_score_algo_1", f1_new_algo_1)

        accuracy_ref = 0.9144385026737968
        precision_ref = 0.9133574007220217
        recall_ref = 0.9693486590038314
        f1_ref = 0.9405204460966543
        tolerance = 0.1

        accuracy_degraded = accuracy_new_algo_1 < accuracy_ref * (1 - tolerance)
        recall_degraded = precision_new_algo_1 < recall_ref * (1 - tolerance)
        precision_degraded = recall_new_algo_1 < precision_ref * (1 - tolerance)
        f1_degraded = f1_new_algo_1 < f1_ref * (1 - tolerance)
    
        # Condition de dégradation pour déclencher le réentraînement
        if accuracy_degraded or recall_degraded or precision_degraded or f1_degraded:
            send_alert("Performances du modèle moins bonnes que les précédentes. Le modèle n'est pas loggé")
    
        else:
            # Log du modèle entraîné
            mlflow.sklearn.log_model(new_rfc_1, "new_rfc_1")
            mlflow.log_param("n_estimators_rfc_1", n_estimators_algo_1)
            mlflow.log_param("max_depth_algo_1", max_depth_algo_1)
            print('Nouveau rfc_1 loggé dans MLFlow')
            print(f"Run ID Algo_1: {run_algo_1.info.run_id}")

        # ENLEVER LES SEGMENTS UTILISES DES TABLES MONITORING
            #delete_training_data_1()  # Suppression des données après réentrainement
    
    return {"message": "Modèle RFC 1 réentraîné avec succès", }


@app.post("/train_algo_2")
async def train():
    async with httpx.AsyncClient() as client:
        # Récupérer les données de la table /data/comp
        try:
            response_comp = await client.get("http://127.0.0.1:8001/data/comp")  
            response_comp.raise_for_status()  
            data_comp = response_comp.json()  # on renvoie les données en json
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données de /data/comp: {e}")


    df_comp = pd.DataFrame(data_comp)
    print("nb lignes dans df_comp :", len(df_comp))
    
    df_comp = df_comp.loc[df_comp['feedback_user'] == 'agree']

    array_comp = df_comp.to_numpy()

    X_algo_2 =  np.array(df_comp['embedding'].to_list())

    array_comp = np.concatenate([array_comp, X_algo_2], axis=1)
    print("array_comp : ", array_comp.shape)

        # PREPA DATA ALGO 2
    Y_algo_2 = array_comp[:, 3]
    X_algo_2 = X_algo_2.astype(float)
    Y_algo_2 = Y_algo_2.astype(int)

    X_train_algo_2, X_test_algo_2, y_train_algo_2, y_test_algo_2 = train_test_split(X_algo_2, Y_algo_2, test_size=0.20, random_state=42, stratify=Y_algo_2)

    
    # IMPORTER les fonctions de RECHERCHE D'HYPERPARAMETRES PUIS D'ENTRAINEMENT
    n_estimators_algo_2, max_depth_algo_2 = search_hyperparam(X_train_algo_2, y_train_algo_2)

    accuracy_new_algo_2, precision_new_algo_2, recall_new_algo_2, f1_new_algo_2, new_rfc_2 = model_rfc(X_train_algo_2, y_train_algo_2, 
                                                                                                       n_estimators_algo_2, max_depth_algo_2, 
                                                                                                       X_test_algo_2, y_test_algo_2)

    # TESTER LES PERFORMANCES DES NOUVEAUX MODELES
    with mlflow.start_run(run_name='Suivi Algo_2') as run_algo_2 :
    # Log des métriques pour rfc_2
        mlflow.log_metric("accuracy_algo_2", accuracy_new_algo_2)
        mlflow.log_metric("precision_algo_2", precision_new_algo_2)
        mlflow.log_metric("recall_algo_2", recall_new_algo_2)
        mlflow.log_metric("f1_score_algo_2", f1_new_algo_2)

        accuracy_ref = 0.9144385026737968
        precision_ref = 0.9133574007220217
        recall_ref = 0.9693486590038314
        f1_ref = 0.9405204460966543
        tolerance = 0.1

        accuracy_degraded = accuracy_new_algo_2 < accuracy_ref * (1 - tolerance)
        recall_degraded = precision_new_algo_2 < recall_ref * (1 - tolerance)
        precision_degraded = recall_new_algo_2 < precision_ref * (1 - tolerance)
        f1_degraded = f1_new_algo_2 < f1_ref * (1 - tolerance)
    
        # Condition de dégradation pour déclencher le réentraînement
        if accuracy_degraded or recall_degraded or precision_degraded or f1_degraded:
            send_alert("Performances du modèle moins bonnes que les précédentes. Le modèle n'est pas loggé")

        else:
            # Log du modèle entraîné
            mlflow.sklearn.log_model(new_rfc_2, "new_rfc_2")
            mlflow.log_param("n_estimators_rfc_2", n_estimators_algo_2)
            mlflow.log_param("max_depth_algo_2", max_depth_algo_2)
            print('Nouveau rfc_2 loggé dans MLFlow')
            print(f"Run ID Algo_2: {run_algo_2.info.run_id}")

        # ENLEVER LES SEGMENTS UTILISES DES TABLES MONITORING
            #delete_training_data_2()  # Suppression des données après réentrainement
    
        return {"message": "Modèle RFC 2 réentraîné avec succès", }
    

    

