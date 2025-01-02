import requests
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd

# REPRENDRE L'AUTOMATISATION DE L'EXTRACTION + LIENS AVEC table_offres

# identifiants client France Travail
CLIENT_ID = 'PAR_matchingesco_dcf45d374cd576dc2fddf2fcf68acbd09cc22cb453e770714c24bd1b0ee87557'
CLIENT_SECRET = 'bb844e4e645100d85efec1b83fa1e30e900f99692f55278873289f1aaa4affb4'
TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire" 
REGION_CODE = 53
MAX_OFFRES = 50
ENDPOINT = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

#Date actualisée chaque jour
date_format = "%Y-%m-%dT%H:%M:%SZ"
# Date actuelle (aujourd'hui) au format spécifié
maxCreationDat = datetime.utcnow().strftime(date_format)
# Date d'hier (aujourd'hui - 1 jour) au format spécifié
minCreationDat = (datetime.utcnow() - timedelta(days=1)).strftime(date_format)

results_day = []

def get_access_token():
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'client_credentials',  # spécifie le type d'accès
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'api_offresdemploiv2 o2dsoffre'
    }
    
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        print("Jeton d'accès obtenu:", access_token)
        return access_token
    else:
        print("Erreur lors de l'obtention du jeton d'accès:", response.status_code, response.text)
        return None


def fetch_offres_bretagne():
    # Obtenir le jeton d'accès
   
    access_token = get_access_token()

    if not access_token:
        return "Impossible de récupérer le jeton d'accès. Abandon de la requête."
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    params = {
        'region': REGION_CODE,
        'range': f'0-{150 - 1}',                #Pagination des données. La plage de résultats est limitée à 150.
        'maxCreationDate' : maxCreationDat,
        'minCreationDate': minCreationDat
    }

    # Requête pour récupérer les offres
    response = requests.get(ENDPOINT, headers=headers, params=params)
    if response.status_code >= 200 and response.status_code < 300 :
        data = response.json()
        offres =  data["resultats"]
        print(f"[{datetime.now()}] Extraction réussie : {len(offres)} offres récupérées.")

        for offre in offres:
            results_day.append({'id' : offre['id'],
                'intitule' : offre['intitule'],
                'description' : offre['description'],
                'date_actu' : offre['dateActualisation'],
                'lieu_travail' : offre['lieuTravail'],
                'code_rome' : offre['romeCode']
            })    

        return pd.DataFrame(results_day)
    else:
        print(f"[{datetime.now()}] Échec de l'extraction. Code erreur : {response.status_code}, {response.text}")


# Planifier l'exécution une fois par semaine
#schedule.every().day.at("15:00").do(fetch_offres_bretagne)

# Lancer la planification
#while True:
#    schedule.run_pending()
#    time.sleep(1)
