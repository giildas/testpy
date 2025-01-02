import pytest
import sys
import os
from unittest.mock import patch
import requests

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from functions import calcul_metrics
from functions import calcul_metrics as calc_metrics


# Patch des requetes serveurs avant l'importation de app.py
@patch('requests.get')
def test_example(mock_get):
    # Simule une réponse de type "200 OK"
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = 'OK'

    # Teste la fonction qui effectue la requête HTTP
    response = requests.get('http://127.0.0.1:5001/api/test')

    # Vérifie la réponse
    assert response.status_code == 200
    assert response.text == 'OK'

def test_calcul_metrics():
    # Données factices pour le test
    y_true_contxt = [(1, 'agree'), (2, 'disagree'), (3, 'agree'), (4, 'agree')]  # Valeurs réelles avec feedback
    y_pred_contxt = [(1, 1), (2, 0), (3, 1), (4, 0)]

    # Appel de la fonction calcul_metrics
    accuracy_contxt, precision_contxt, recall_contxt, f1_contxt = calc_metrics(y_true_contxt, y_pred_contxt)

    # Vérifier que la fonction retourne bien quatre valeurs
    assert isinstance(accuracy_contxt, float), "accuracy_contxt doit être un float"
    assert isinstance(precision_contxt, float), "precision_contxt doit être un float"
    assert isinstance(recall_contxt, float), "recall_contxt doit être un float"
    assert isinstance(f1_contxt, float), "f1_contxt doit être un float"

    # Vous pouvez aussi vérifier que les valeurs ne sont pas NaN ou infinies, si vous le souhaitez
    assert accuracy_contxt >= 0 and accuracy_contxt <= 1, "accuracy_contxt doit être entre 0 et 1"
    assert precision_contxt >= 0 and precision_contxt <= 1, "precision_contxt doit être entre 0 et 1"
    assert recall_contxt >= 0 and recall_contxt <= 1, "recall_contxt doit être entre 0 et 1"
    assert f1_contxt >= 0 and f1_contxt <= 1, "f1_contxt doit être entre 0 et 1"

@patch('numpy.load')
def test_best_skill(mock_load):

    dtype = [('id', 'i4'), ('skill', 'U10')] + [('vector_' + str(i), 'f4') for i in range(1024)]
    mock_array = np.zeros((10,), dtype=dtype)

    # Remplir les données avec des valeurs factices
    for i in range(10):
        mock_array[i]['id'] = i
        mock_array[i]['skill'] = f'Skill{i}'  # Définir des compétences comme Skill0, Skill1, ...
        for j in range(1024):
            mock_array[i][f'vector_{j}'] = np.random.rand()  # Valeurs aléatoires pour les vecteurs

    mock_load.return_value = mock_array

    # Définir un vecteur fictif pour tester best_skill
    test_vector = np.array([mock_array[0][f'vector_{j}'] for j in range(1024)])

    assert test_vector.shape == (1024,), "La dimension du vecteur n'est pas 1024"

    vector_dim = 1024
    index = faiss.IndexFlatL2(vector_dim)
    vectors_comp = np.array([
        [mock_array[i][f'vector_{j}'] for j in range(1024)]
        for i in range(10)
    ])

    assert vectors_comp.shape == (10, 1024), "La forme de vectors_comp n'est pas correcte"

    index.add(vectors_comp)

    with patch('functions.index', index), patch('functions.vectors_comp', vectors_comp):
    # Appeler la fonction best_skill
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from functions import best_skill

        threshold_skill=0
        result = best_skill(test_vector, threshold_skill=threshold_skill)

    # Vérifier que le résultat est correct (basé sur la logique de ta fonction)
    assert isinstance(result, dict), "Le résultat doit être un dictionnaire"


# Fixture client déplacée en dehors de la fonction de test
@pytest.fixture
def client():
    # Ajout du répertoire parent au sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Importation de l'application Flask après avoir mocké mlflow
    from app import app
    with app.test_client() as client:
        yield client
