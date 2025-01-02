# Lancer FastAPI avec uvicorn : uvicorn api_gha:app --host 127.0.0.1 --port 8000
# Lancer MLFlow sur le port 5001 : mlflow ui --port 5001 / mlflow server --host 127.0.0.1 --port 5001
# Lancer l'appli app_nov.py sur le port 5000

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import asyncio
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import os
import subprocess

from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import mlflow
import mlflow.sklearn

from functions import calcul_metrics, check_performance_degradation, send_alert, get_metrics, send_train_request
from models_sql import Base, ImportSegmentContxt, ImportSegmentComp, User

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

app = Flask(__name__)
secret_key = os.urandom(24)  # A CHANGER : Utiliser une clé fixe pour la production
app.config['SECRET_KEY'] = secret_key
#app.secret_key = secret_key  


# SQL:
DATABASE_URL = "postgresql://postgres:postgre@localhost/db_pco"
engine = create_engine(DATABASE_URL)
DBSession = sessionmaker(bind=engine)
db_session = DBSession()


# Configurer le suivi avec MLFLOW sur un port différent
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Utiliser MLFlow sur le port 5001
mlflow.set_experiment("Monitoring des perf des modèles PCO")  # Nom de l'expérience dans MLFlow

# Modèle vectorisation segments
model_embed = SentenceTransformer("Lajavaness/sentence-camembert-large")


# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Préparer les données pour la requête
        user_data = {
            "username": username,
            "password": password
        }

        # Envoyer les données à la route /users d'api_data.py
        try:
            response = requests.post("http://127.0.0.1:8001/users", json=user_data)
            if response.status_code == 200:
                flash('Registration successful. Please login.')
                return redirect(url_for('login'))
            elif response.status_code == 400:
                flash(response.json().get("detail", "Username already exists. Please choose a different one."))
                return redirect(url_for('register'))
            else:
                flash("An unexpected error occurred.")
                return redirect(url_for('register'))
        except requests.exceptions.RequestException as e:
            flash(f"Could not connect to the user registration service: {str(e)}")
            return redirect(url_for('register'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verify if the user exists and the password is correct
        user = db_session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            session['username'] = username
            flash('Login successful!')
            return redirect(url_for('index'))

        flash('Invalid username or password. Please try again.')
        return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Home route where the input form is displayed (requires login)
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        flash('You need to be logged in to access this page.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        input_text = request.form['text_input']
        consent = request.form.get('consent')
        if not consent:
            flash('Vous devez accepter la politique de confidentialité pour continuer.')
            return redirect(url_for('index'))

        # API endpoint details
        url = 'http://127.0.0.1:8000/predict'
        params = {'text_input': input_text}
        headers = {'accept': 'application/json'}

        # Call the API + réponses
        response = requests.post(url, params=params, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            session['segmented_text'] = response_json.get('segmented_text', [])
            session['results_algo_1'] = response_json.get('results_algo_1', [])
            session['results_algo_2'] = response_json.get('results_algo_2', [])
            session['results_comp'] = response_json.get('results_comp', [])
            session['nb_segm_algo_1'] = response_json.get('nombre_segments_algo_1', [])
            session['nb_segm_algo_2'] = response_json.get('nombre_segments_algo_2', [])

            return redirect(url_for('feedbacks'))

    return render_template('index.html')

@app.route('/feedbacks', methods=['GET', 'POST'])
def feedbacks():
    if 'username' not in session:
        flash('You need to be logged in to access this page.')
        return redirect(url_for('login'))

    segmented_text = session.get('segmented_text', [])
    results_algo_1 = session.get('results_algo_1', [])
    results_algo_2 = session.get('results_algo_2', [])

    # Préparer les résultats pour le feedback
    session['feedback_results_contxt'] = [{'segment': result['segment'], 'contxt_predict': result['offre_predict'], 'feedback': None} for result in results_algo_1]
    session['feedback_results'] = [{'segment': result['segment'], 'comp_predict': result['comp_predict'], 'feedback': None} for result in results_algo_2]
    print("feedbacks avant envoi :", session['feedback_results_contxt'])

    flash("Merci pour vos feedbacks !")
    return render_template('feedbacks.html',
                           segmented_text=segmented_text,
                           results_algo_1=results_algo_1,
                           results_algo_2=results_algo_2)


@app.route('/handle_user_feedback', methods=['POST'])
def handle_user_feedback():
    '''if 'username' not in session:
        flash('You need to be logged in to provide feedback.')
        return redirect(url_for('login'))'''

    y_pred_contxt = []
    y_true_contxt = []
    y_pred_offre = []
    y_true_offre = []
    count_segments = 0
    count_segments_feedbackes = 0

    username = session['username']
    print('username :', username)

    # Récupérer les résultats de la session
    feedback_results_contxt = session.get('feedback_results_contxt', [])
    feedback_results = session.get('feedback_results', [])
    print("feedback_results avant mise à jour :")
    print(feedback_results_contxt)

    # On récupère les prédictions avant mise à jour : ALGO 1 y_pred_contxt
    for idx, item in enumerate(feedback_results_contxt):
        y_1 = (idx, item['contxt_predict'])
        y_pred_contxt.append(y_1)
    print("y_pred_contxt : ", y_pred_contxt)
    # On récupère les prédictions avant mise à jour : ALGO 2 y_pred_offre
    for idx, item in enumerate(feedback_results):
        y_1 = (idx, item['comp_predict'])
        y_pred_offre.append(y_1)
    print("y_pred_offre : ", y_pred_offre)

    # Récupérer le nbre de segments pour chaque algo
    nb_segm_algo_1 = session['nb_segm_algo_1']
    nb_segm_algo_2 = session['nb_segm_algo_2']

    # ALGO 1 : parcourir chaque feedback soumis dans le formulaire
    for idx, item in enumerate(feedback_results_contxt):
        feedback = request.form.get(f'feedback_{idx}')  # Récupérer le feedback pour chaque segment  
        item['feedback'] = feedback  # Mettre à jour le feedback dans la session

        # On récupère les prédictions avant mise à jour : y_true_offre
        y_2 = (idx, item['feedback'])
        y_true_contxt.append(y_2)

        if feedback:
            item['feedback'] = feedback  # Mettre à jour le feedback dans la session
    print("y_true_contxt : ", y_true_contxt)       

    # ALGO 2 : Parcourir chaque feedback soumis dans le formulaire
    for idx, item in enumerate(feedback_results):
        feedback = request.form.get(f'feedback_{idx}')  # Récupérer le feedback pour chaque segment
        count_segments += 1   
        item['feedback'] = feedback  # Mettre à jour le feedback dans la session

        # On récupère les prédictions avant mise à jour : y_true_offre
        y_2 = (idx, item['feedback'])
        y_true_offre.append(y_2)

        if feedback:
            item['feedback'] = feedback  # Mettre à jour le feedback dans la session
            count_segments_feedbackes += 1

    # Sauvegarder les feedbacks mis à jour dans la session
    session['feedback_results_contxt'] = feedback_results_contxt
    session['feedback_results'] = feedback_results
    print('feedback_results_contxt: ', session['feedback_results_contxt'])

    # INVERSER LES PREDICTIONS QUAND FEEDBACKS == 'disagree' avant d'importer les segments dans les tables

    # Importation dans db_pco
    try:
        for item in feedback_results_contxt:
            if item['feedback'] is not None:
                embedding = model_embed.encode(item['segment']).tolist()

                new_record = ImportSegmentContxt(
                ref_user=username,
                segment=item['segment'],
                prediction_contxt=item['contxt_predict'],
                feedback_user=item['feedback'],
                embedding = embedding
                )
                db_session.add(new_record)
            db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error inserting into database: {e}")

    try:
        for item in feedback_results:
            if item['feedback'] is not None:
                embedding = model_embed.encode(item['segment']).tolist()

                new_record = ImportSegmentComp(
                ref_user=username,
                segment=item['segment'],
                prediction=item['comp_predict'],
                feedback_user=item['feedback'],
                embedding = embedding
                )
                db_session.add(new_record)
            db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error inserting into database: {e}")
    finally:
        db_session.close()


    # CALCUL DES METRIQUES POUR L'ENVOYER A l'experiment MLFLOW : fonction calcul_metrics
    accuracy_contxt, precision_contxt, recall_contxt, f1_contxt = calcul_metrics(y_true_contxt, y_pred_contxt)
    accuracy_comp, precision_comp, recall_comp, f1_comp = calcul_metrics(y_true_offre, y_pred_offre)


    # on lance le run maintenant:
    # on récupére le nbre de runs dans l'experiment MLFlow
    experiment_name = "Monitoring des perf des modèles PCO"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"L'expérimentation '{experiment_name}' est introuvable.")
    else:
        print("Expérimentation trouvée.")

    run_name = username
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    print('runs :', runs)
    num_runs = len(runs)
    print("len de runs :", num_runs)
    run_index = num_runs

    with mlflow.start_run(run_name=run_name) :
        mlflow.log_param("run_index", run_index)
        mlflow.log_param("nb_segm_algo_1", nb_segm_algo_1)
        mlflow.log_param("nb_segm_algo_2", nb_segm_algo_2)
        mlflow.log_metric("accuracy_contxt", accuracy_contxt)
        mlflow.log_metric("precision_contxt", precision_contxt)
        mlflow.log_metric("recall_contxt", recall_contxt)
        mlflow.log_metric("f1_score_contxt", f1_contxt)
        mlflow.log_metric("accuracy_comp", accuracy_comp)
        mlflow.log_metric("precision_comp", precision_comp)
        mlflow.log_metric("recall_comp", recall_comp)
        mlflow.log_metric("f1_score_comp", f1_comp)
    
    # CALCUL DU NBRE DE RUNS ET CHECK PERFORMANCES DES 2 ALGOS 
    if num_runs % 2 != 0 :
        print(f"Nombre de runs pour l'expérimentation '{experiment_name}': {num_runs}")
        return redirect(url_for('results'))
        
    elif num_runs % 2 == 0 and num_runs != 0:
        print(f"Nombre de runs atteint un multiple de 10 : {num_runs}")
            
        latest_runs, metrics_10runs = get_metrics(runs)
        print('metrics_10runs : ', metrics_10runs)
        metrics_1 = [(run["accuracy_contxt"], run["precision_contxt"], run["recall_contxt"], run["f1_score_contxt"])
                    for run in metrics_10runs]
        metrics_2 = [(run["accuracy_comp"], run["precision_comp"], run["recall_comp"], run["f1_score_comp"])
                    for run in metrics_10runs]
        print("métriques_algo_1 des 10 runs précédents :", metrics_1)
        print("métriques_algo_1 des 10 runs précédents :", metrics_2)

        stable_algo_1 = check_performance_degradation(metrics_1)
        stable_algo_2 = check_performance_degradation(metrics_2)

        if stable_algo_1:
            if stable_algo_2:
                return redirect(url_for('results')) # changer pour la route résults ?
            
            if not stable_algo_2:
                send_alert("Performances dégradées de rfc_2")
                return redirect(url_for('results'))

                '''response_algo_2 = asyncio.run(send_train_request('http://127.0.0.1:8000/train_algo_2'))
                if response_algo_2.status_code == 200:
                    return redirect(url_for('results'))
                else:
                # Gérer les erreurs ici si nécessaire
                    print('message : rfc_1 stable ; Erreur lors du réentraînement de rfc_2')
                    return redirect(url_for('results'))'''
            
        if not stable_algo_1:
            send_alert("Performances dégradés de rfc_1")

            #response_algo_1 = asyncio.run(send_train_request('http://127.0.0.1:8000/train_algo_1'))
            #if response_algo_1.status_code == 200:
            if stable_algo_2:
                print('message : rfc_1 doit être réentrainé ; rfc_2 stable')
                return redirect(url_for('results'))
        
            if not stable_algo_2:
                send_alert("Performances dégradés de rfc_1 et rfc_2")
                return redirect(url_for('results'))
                '''response_algo_2 = asyncio.run(send_train_request('http://127.0.0.1:8000/train_algo_2'))
                    if response_algo_2.status_code == 200:
                        return redirect(url_for('results'))
                    else:
                        print('message : rfc_1 et rfc_2 doivent être réentrainés')
                        return redirect(url_for('results'))'''
        


# ROUTE /results
@app.route('/results', methods=['GET', 'POST'])
def results():
    print(session['username'])
    if 'username' not in session:
        flash('You need to be logged in to view the results.')
        return redirect(url_for('login'))
    
    # Récupérer les résultats de la session ou d'une source externe
    results_comp = session.get('results_comp', [])

    return render_template('results.html', 
                           results_comp=results_comp
                           )


if __name__ == '__main__':
    app.run(debug=True)
