import numpy as np
import pandas as pd

import spacy
from spacy.lang.fr.examples import sentences
from spacy.lang.fr import French

import spacy
from spacy.lang.fr.examples import sentences
from spacy.lang.fr import French

import re

import httpx

import faiss
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow

import os

import requests

# FONCTIONS POUR SEGMENTATION
# Initialisation du modèle nlp : https://spacy.io/models/fr
nlp = spacy.load("fr_dep_news_trf") # modèle fr_core_news_sm + léger
# Initialisation du modèle de langue
model = SentenceTransformer("Lajavaness/sentence-camembert-large")

# Fonction pour ajouter de la ponctuation si l'offre n'en a pas
def correct_maj_ds_mot(text):
    # Expression régulière pour trouver une majuscule non précédée par un espace ou une ponctuation
    corrected_text = re.sub(r'(?<=[a-zéèêàùôûîï])(?=[A-Z])', '. ', text)
    return corrected_text

# Fonction pour uniformiser la ponctuation et la forme de chaque offre en amont de la segmentation
# but : paragraphe clean avec ponctuation 

def clean_text(text):
    text = correct_maj_ds_mot(text)
    text = text.replace('¿', '')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s*([.,;:!?\(\)])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Fonction pour segmenter l'offre sur la base de la ligne
def split_on_newlines(doc):
    lines = []
    for sent in doc.sents:
        sent_text = re.sub(r'[.;:](?![.]{2})', '\n', sent.text)
        sent_text = sent_text.replace('),', ')\n')
        
        # Nettoyage du texte pour enlever les caractères indésirables
        sent_text = re.sub(r'[^\w\s\'\(\)\%\+/]', ' ', sent_text)
        
        # Créer une variable doc pour appliquer NER et segmentation par verbes
        doc_segment = nlp(sent_text)
        
        modified_segment = []
        verb_count = 0
        ignore_verbs = 0  # Compteur pour ignorer certains verbes
        for i, token in enumerate(doc_segment):
            if token.pos_ == 'VERB':
                verb_count += 1
                
                # Vérifier si le verbe est utilisé comme adjectif ou participe présent
                if token.tag_ in ['VPP', 'VPR', 'ADJ'] or token.dep_ in ['amod', 'acl']:
                    ignore_verbs += 1
                
                if verb_count - ignore_verbs == 2:  # Segmentation si 2e verbe non ignoré
                    # Vérifier s'il y a "et" entre les deux verbes
                    if not (i > 1 and doc_segment[i-1].text.lower() == "et" and doc_segment[i-2].pos_ == 'VERB'):
                        if i > 0 and doc_segment[i-1].text.endswith("'"):
                            # Ajouter le saut de ligne avant le token qui précède l'apostrophe
                            modified_segment.insert(-1, '\n')
                        else:
                            modified_segment.append('\n')
                    verb_count = 0  # Réinitialiser le compteur de verbes
                    ignore_verbs = 0  # Réinitialiser le compteur des verbes ignorés
            
            # Ajouter la condition pour le caractère "+"
            if token.text == "+":
                modified_segment.append('\n')
            
            modified_segment.append(token.text)
        
        # Joindre les tokens pour former la phrase segmentée
        sent_text_final = ' '.join(modified_segment)
    
        # Séparation par des sauts de ligne simples
        segments = sent_text_final.split('\n')
        for segment in segments:
            segment = segment.strip()
            
            if len(segment) == 1 or not segment or re.match(r'^[^\w\s]+$', segment):
                continue

            lines.append(segment)
    
    return lines

# Fonction finale à appliquer sur une offre
def lignes_segm(cellule):
    cellule_clean = clean_text(cellule)
    doc = nlp(cellule_clean)
    return list(split_on_newlines(doc))


# FONCTION DE SIMILARITE
'''is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'


if is_github_actions:
    # Si c'est GitHub Actions, utilisez un chemin relatif
    file_path = os.path.join(os.path.dirname(__file__), 'DB_pco', 'array_comp_esco.npy')
else:'''
# Sinon, utilisez un chemin absolu (pour votre machine locale)
file_path = 'C:\\Users\\Utilisateur\\Documents\\Prepa_Diplome\\PCO_dec2\\DB_pco\\array_comp_esco.npy'

# Charger le fichier .npy avec le chemin approprié
array_comp = np.load(file_path, allow_pickle = True)


#array_comp = np.load('C:\\Users\\Utilisateur\\Documents\\Prepa_Diplome\\PCO_nov\\DB_pco\\array_comp_esco.npy', allow_pickle = True)
vectors_comp = array_comp[: , 2:]

vector_dim = 1024
index = faiss.IndexFlatL2(vector_dim)
index.add(vectors_comp)

def best_skill(vector, threshold_skill=0.7):
    
    best_skill = None
    best_score = 0

    # Perform a similarity search
    vector = np.array(vector) # il faut repasser vector en array numpy, car on l'a tranformé en list dans le script de l'api
    distances, indices = index.search(vector[None, :], 5)
    
    for indice in indices[0]:
        
        cos_sim = cosine_similarity(vector.reshape(1, -1), vectors_comp[indice].reshape(1, -1))
        cos_sim = round(cos_sim[0][0], 2)
        #cos_sim = 1 - (distances[indice] ** 0.5)

        print(f"Comparing: cos_sim={cos_sim}, threshold={threshold_skill}, best_score={best_score}")

        if cos_sim >= threshold_skill and cos_sim > best_score:
            best_skill = str(array_comp[:, 1][indice])
            best_score = cos_sim

    if best_skill is not None:
        return {best_skill: best_score}
    else:
        return {}
    

# CALCUL DES METRIQUES
def calcul_metrics(y_true, y_pred):
    # on filtre d'abord les retours None
    y_true_filtered = []
    y_pred_filtered = []

    # Parcourir les deux listes simultanément
    for true_item, pred_item in zip(y_true, y_pred):
        if true_item[1] is not None:  # Si y_true_offre ne contient pas None
            y_true_filtered.append(true_item)
            y_pred_filtered.append(pred_item)

    y_true_final = []

    # Parcourir les deux listes simultanément
    for true_item, pred_item in zip(y_true_filtered, y_pred_filtered):
        index, feedback = true_item
        pred_index, pred_value = pred_item

        if index == pred_index:
            if feedback == 'agree':
                # Remplacer par la prédiction correspondante
                y_true_final.append((index, pred_value))
            elif feedback == 'disagree':
                # Remplacer par l'inverse de la prédiction
                y_true_final.append((index, 1 - pred_value))

    # Résultat final
    print("y_true_final:", y_true_final)

    # Extraire uniquement le deuxième élément (la valeur de prédiction)
    y_true_values = [value for _, value in y_true_final]
    y_pred_values = [value for _, value in y_pred_filtered]
    
    accuracy = accuracy_score(y_true_values, y_pred_values)
    precision = precision_score(y_true_values, y_pred_values)
    recall = recall_score(y_true_values, y_pred_values)
    f1 = f1_score(y_true_values, y_pred_values)

    return accuracy, precision, recall, f1


# MONITORING
# Fonction de récupération des métriques des runs depuis l'experiment MLFlow
def get_metrics(runs):
    sorted_runs = runs.sort_values(by="start_time", ascending=False)  # Tri décroissant par start_time
    latest_runs = sorted_runs.head(10)
    print("latest_runs :", latest_runs)
    metrics_10runs = []

    # Récupération des métriques de chaque run
    for idx, run in latest_runs.iterrows():
        run_id = run["run_id"]
        print(f"\nRun ID: {run_id}")

        # Charger le run pour obtenir ses métriques
        run_data = mlflow.get_run(run_id).data
        metrics = run_data.metrics

        # Affichage des métriques pour ce run
        run_metrics = {"run_id": run_id}  # Créer un dictionnaire pour chaque run
        for metric_name, metric_value in metrics.items():
            print(f" - {metric_name}: {metric_value}")
            run_metrics[metric_name] = metric_value
        metrics_10runs.append(run_metrics)  # Ajouter le dictionnaire du run à la liste principale
    
    return latest_runs, metrics_10runs


# Valeurs de référence pour les métriques
accuracy_ref = 0.9144385026737968
precision_ref = 0.9133574007220217
recall_ref = 0.9693486590038314
f1_ref = 0.9405204460966543
tolerance = 0.1

def check_performance_degradation(metrics):
    avg_accuracy = np.mean([m[0] for m in metrics])
    avg_precision = np.mean([m[1] for m in metrics])
    avg_recall = np.mean([m[2] for m in metrics])
    avg_f1 = np.mean([m[3] for m in metrics])
    
    print(f"Moyennes des 10 derniers runs : Accuracy: {avg_accuracy}, Precision: {avg_precision}, Recall: {avg_recall}, F1-score: {avg_f1}")
    
    accuracy_degraded = avg_accuracy < accuracy_ref * (1 - tolerance)
    recall_degraded = avg_recall < recall_ref * (1 - tolerance)
    precision_degraded = avg_precision < precision_ref * (1 - tolerance)
    f1_degraded = avg_f1 < f1_ref * (1 - tolerance)
    
    # Condition de dégradation pour déclencher le réentraînement
    if accuracy_degraded or recall_degraded or precision_degraded or f1_degraded:
        return False  # Dégradation des performances
    else:
        return True  


# ENVOI ALERTES
#import smtplib  # Pour envoyer des emails

def check_alerts():
    client = mlflow.tracking.MlflowClient()
    runs = client.list_run_infos(experiment_id="")
    
    # Récupérer les métriques les plus récentes
    last_run = runs[-1]  # Dernière exécution
    metrics = client.get_run(last_run.run_id).data.metrics

    if metrics["nombre_prédictions_faible_confiance"] > 0:
        send_alert()  # Fonction pour envoyer un email

def send_alert(message):
    # Exemple de fonction pour envoyer un email d'alerte
    #with smtplib.SMTP('smtp.example.com') as server:
    #    server.login("user@example.com", "password")
    #    message = "Alerte: Des prédictions à faible confiance ont été détectées."
    #    server.sendmail("from@example.com", "to@example.com", message)
    print("ALERTE :", message)


# FONCTION DE REQUETE DES ROUTES D'ENTRAINEMENT
async def send_train_request(url):
    try:
        # Augmenter le timeout à 60 secondes par exemple
        timeout = httpx.Timeout(60.0)  # Timeout de 60 secondes
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url)
        return response
    except httpx.TimeoutException as e:
        print(f"Timeout Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# FONCTION DE SUPPRESSION DES DONNES DE MONITORING SI REENTRAINEMENT ET VALIDATION DES NOUVEAUX MODELES
import psycopg2

def connect_to_postgres():
    try:
        connection = psycopg2.connect(
            dbname="db_pco",
            user="postgres",
            password="potgre",
            host="localhost",  # généralement localhost ou une adresse IP
            port="5432"  # port par défaut de PostgreSQL
        )
        return connection
    except Exception as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None


def delete_training_data_1():
    connection = connect_to_postgres()
    if connection is not None:
        cursor = connection.cursor()
        
        try:
            delete_query_2 = "DELETE FROM table_monitoring_contxt"  # Ajoutez votre condition
            cursor.execute(delete_query_2)
            connection.commit()

            print("Données de réentrainement supprimées avec succès.")

        except Exception as error:
            print(f"Erreur lors de la suppression des données : {error}")
            connection.rollback()  # En cas d'erreur, annulez les changements
        finally:
            cursor.close()
            connection.close()
    else:
        print("Échec de la connexion à la base de données.")

def delete_training_data_2():
    connection = connect_to_postgres()
    if connection is not None:
        cursor = connection.cursor()
        
        try:
            delete_query_1 = "DELETE FROM table_monitoring_comp"
            cursor.execute(delete_query_1)
            connection.commit()

            print("Données de réentrainement supprimées avec succès.")

        except Exception as error:
            print(f"Erreur lors de la suppression des données : {error}")
            connection.rollback()  # En cas d'erreur, annulez les changements
        finally:
            cursor.close()
            connection.close()
    else:
        print("Échec de la connexion à la base de données.")
