# pour changer de port de connexion avec l'API du modèle IA:
# uvicorn connect_db_api:app --host 127.0.0.1 --port 8001

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

app = FastAPI()

DATABASE_CONFIG = {
    "host": "localhost",
    "database": "db_pco",
    "user": "postgres",
    "password": "postgre"
}

# Modèle pour les données de table_offre
class TableOffre(BaseModel):
    id: int
    ref_offre: str
    segment: str
    offre_predict: int

# Modèle pour les données de table_comp
class TableComp(BaseModel):
    id: int
    ref_offre: str
    segment: str
    comp_predict: int

def get_db_connection():
    conn = psycopg2.connect(
        host=DATABASE_CONFIG["host"],
        database=DATABASE_CONFIG["database"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"]
    )
    return conn

@app.get("/load_tables/")
async def load_tables():
    conn = get_db_connection()
    
    # Charger la table_offre
    offre_query = "SELECT * FROM table_contxt"
    table_offre = pd.read_sql_query(offre_query, conn)
    
    # Charger la table_comp
    comp_query = "SELECT * FROM table_comp"
    table_comp = pd.read_sql_query(comp_query, conn)

    conn.close()

    # Convertir les résultats en DataFrame Pandas
    df_algo_1 = pd.DataFrame(table_offre)
    df_algo_2 = pd.DataFrame(table_comp)

    # Supprimer les doublons
    df_algo_1.drop_duplicates(subset='segment', inplace=True)
    df_algo_2.drop_duplicates(subset='segment', inplace=True)

    # Réinitialiser l'index
    df_algo_1.reset_index(drop=True, inplace=True)
    df_algo_2.reset_index(drop=True, inplace=True)

    # Afficher les premières lignes des DataFrames
    print(df_algo_1.head())
    print(df_algo_2.head())

    # Chargement des vecteurs d'embedding
    model = SentenceTransformer("Lajavaness/sentence-camembert-large")

    # Transformation des segments en vecteurs
    segments_algo = np.array(df_algo_1['segment'])
    segments_comp = np.array(df_algo_2['segment'])

    vectors_algo = model.encode(segments_algo, show_progress_bar=True)
    vectors_comp = model.encode(segments_comp, show_progress_bar=True)

    # Concaténation des données avec les vecteurs d'embedding
    array_algo = df_algo_1.to_numpy()
    array_algo = np.concatenate((array_algo, vectors_algo), axis=1)
    
    array_comp = df_algo_2.to_numpy()
    array_comp = np.concatenate((array_comp, vectors_comp), axis=1)

    # Sauvegarde des données en format .npy
    #np.save("array_algo_1.npy", array_algo)
    #np.save("array_algo_2.npy", array_comp)

    return {"message": "Données pour l'entrainement prêtes !"}

# CONDITION : Tant que accuracy(y_test, y_pred_algo_1) et accuracy(y_test, y_pred_algo_2) ne se dégrade pas:
# on ajoute les segments de l'offre traitée dans table_comp : REPRENDRE LA FONCTION load_tables()
'''@app.get("/update_array_comp/")
async def update_array_comp():
    conn = get_db_connection()
    
    # Charger la table_comp
    comp_query = "SELECT * FROM table_comp"
    table_comp = pd.read_sql_query(comp_query, conn)

    conn.close()

    # Convertir les résultats en DataFrame Pandas
    df_algo_2 = pd.DataFrame(table_comp)

    # Supprimer les doublons
    df_algo_2.drop_duplicates(subset='segment', inplace=True)

    # Réinitialiser l'index
    df_algo_2.reset_index(drop=True, inplace=True)

    # Afficher les premières lignes des DataFrames
    print(df_algo_2.head())

    # Chargement des vecteurs d'embedding
    model = SentenceTransformer("Lajavaness/sentence-camembert-large")

    # Transformation des segments en vecteurs
    segments_comp = np.array(df_algo_2['segment'])
    vectors_comp = model.encode(segments_comp, show_progress_bar=True)

    # Concaténation des données avec les vecteurs d'embedding
    array_comp = df_algo_2.to_numpy()
    array_comp = np.concatenate((array_comp, vectors_comp), axis=1)

    # Optionnel : Sauvegarde des données en format .npy
    # np.save("array_comp_updated.npy", array_comp)

    return {"array_comp": array_comp.tolist()}  # Convertir en liste pour JSON'''
