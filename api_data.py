# Pour lancer API DATA : uvicorn api_data:app --reload --host 127.0.0.1 --port 8001

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError, jwt

import requests

import pandas as pd
from datetime import datetime, timedelta

from pydantic import BaseModel

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models_sql import ImportSegmentContxt, ImportSegmentComp, OffresExtract, User
from sqlalchemy.exc import SQLAlchemyError

from werkzeug.security import generate_password_hash

# Utilisez la configuration et la session déjà définies
DATABASE_URL = "postgresql://postgres:postgre@localhost/db_pco"
engine = create_engine(DATABASE_URL)
DBSession = sessionmaker(bind=engine)
db_session = DBSession()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# Clé secrète et algo de signature pour JWT
SECRET_KEY = "1234"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Durée de validité du token

# Modèle Pydantic pour représenter les informations du Token
class Token(BaseModel):
    access_token: str
    token_type: str

# Modèle pour la création d'un utilisateur
class UserCreate(BaseModel):
    username: str
    password: str

# Fonction pour obtenir un utilisateur depuis la base de données
def get_user(username: str):
    return db_session.query(User).filter(User.username == username).first()

# Fonction pour extraire l'utilisateur actuel à partir du token JWT
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Décoder le token JWT pour extraire les informations de l'utilisateur
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user


# Configuration API
app = FastAPI()


# ROUTE /france_travail : extraction des offres d'emploi depuis API Offres d'emploi de France travail
# Identifiants client France Travail
CLIENT_ID = 'PAR_matchingesco_dcf45d374cd576dc2fddf2fcf68acbd09cc22cb453e770714c24bd1b0ee87557'
CLIENT_SECRET = 'bb844e4e645100d85efec1b83fa1e30e900f99692f55278873289f1aaa4affb4'
TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
ENDPOINT = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
REGION_CODE = 53

# Format de date
date_format = "%Y-%m-%dT%H:%M:%SZ"

# Fonction pour obtenir un jeton d'accès
def get_access_token():
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'api_offresdemploiv2 o2dsoffre'
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise HTTPException(status_code=500, detail=f"Erreur d'authentification : {response.text}")

# Fonction pour récupérer les offres
def fetch_offres_bretagne():
    # Calcul des dates
    max_creation_date = datetime.utcnow().strftime(date_format)
    min_creation_date = (datetime.utcnow() - timedelta(days=1)).strftime(date_format)

    # Obtenir le jeton d'accès
    access_token = get_access_token()

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    params = {
        'region': REGION_CODE,
        'range': f'0-{150 - 1}',  # Limité à 150 offres
        'maxCreationDate': max_creation_date,
        'minCreationDate': min_creation_date
    }

    response = requests.get(ENDPOINT, headers=headers, params=params)
    if response.status_code >= 200 and response.status_code < 300:
        data = response.json()
        offres = data.get("resultats", [])
        results = [
            {
                'id': offre['id'],
                'intitule': offre['intitule'],
                'description': offre['description'],
                'date_actu': offre['dateActualisation'],
                'lieu_travail': offre['lieuTravail'],
                'code_rome': offre['romeCode']
            }
            for offre in offres
        ]
        return pd.DataFrame(results)
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Erreur d'extraction : {response.text}")

# Route API pour lancer le scraping
@app.get("/france_travail")
def get_offres_bretagne(current_user: User = Depends(get_current_user)):
    try:
        df = fetch_offres_bretagne()

        # Planifier l'exécution une fois par semaine
        #schedule.every().day.at("15:00").do(fetch_offres_bretagne)

        # Lancer la planification
        #while True:
        #    schedule.run_pending()
        #    time.sleep(1)

        return df.to_dict(orient="records")  # Retourne les données sous forme de liste de dictionnaires
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ROUTE POUR PREPARATION TABLE OFFRES_EMPLOI
@app.get("/offres")
def get_combined_offres():
    try:
        # on récupère les offres de pole emploi
        df_pole_emploi = fetch_offres_bretagne()
        df_pole_emploi = df_pole_emploi[["id", "intitule", "description"]]
        df_pole_emploi = df_pole_emploi.dropna()
        df_pole_emploi = df_pole_emploi.rename(columns={
                    "id": "id_offre",
                    })

        # on charge les offres scrapées
        csv_file = "C:\\Users\\Utilisateur\\Documents\\Prepa_Diplome\\PCO_nov\\DB_pco\\rennes_metropole.csv"
        try:
            df_rennes_metro = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Fichier {csv_file} introuvable.")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Erreur lors du chargement du fichier CSV : {str(e)}")
        
        df_rennes_metro = df_rennes_metro[['ref_offre', 'intitule_poste', 'descri_mission']]
        df_rennes_metro = df_rennes_metro.dropna()
        df_rennes_metro = df_rennes_metro.rename(columns={
                    "ref_offre": "id_offre",
                    "intitule_poste": "intitule",
                    "descri_mission": "description"
                    })
        
        df_offres = pd.concat([df_pole_emploi, df_rennes_metro], axis=0)

        df_offres.reset_index(drop=True, inplace=True)
        print(df_offres.head(1))

        try:
            for _, row in df_offres.iterrows():
                offre = OffresExtract(
                    id_offre=row['id_offre'],    
                    intitule=row['intitule'],
                    description=row['description']
                )
                db_session.add(offre)

            db_session.commit()
        except SQLAlchemyError as e:
            db_session.rollback() 
            raise HTTPException(status_code=500, detail=f"Erreur d'insertion SQL : {str(e)}")

        return {"message": "Données insérées avec succès dans la table offres_extract."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ROUTE POUR GERER L'ENREGISTREMENT DES NVO UTILISATEURS DANS users DE db_pco
# Route pour obtenir un token JWT
@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(username=form_data.username)
    if user is None or not user.check_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Créez un token JWT pour l'utilisateur
    access_token = user.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}



# Route pour ajouter un utilisateur
@app.post("/users")
def create_user(user_data: UserCreate):
    username = user_data.username
    password = user_data.password
    
    # Vérifier si l'utilisateur existe déjà
    existing_user = db_session.query(User).filter_by(username=username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Créer un nouvel utilisateur et hacher le mot de passe
    new_user = User(username=username)
    new_user.set_password(password)
    
    # Ajouter l'utilisateur à la base de données
    db_session.add(new_user)
    db_session.commit()
    db_session.refresh(new_user)
    
    return {"message": "User created successfully", "user_id": new_user.id}


# ROUTES POUR IMPORTER LES TABLES DE MONITORING : REENTRAINEMENT DES MODELES
# Route pour obtenir les données de `ImportSegmentComp`
@app.get("/data/comp")
def get_table_comp():
    # Récupération des données de la table ImportSegmentComp
    comp_data = db_session.query(ImportSegmentComp).all()
    if not comp_data:
        raise HTTPException(status_code=404, detail="Table comp vide")
    
    return [{"id": item.id,
             "ref_user": item.ref_user,
             "segment": item.segment,
             "prediction": item.prediction,
             "feedback_user": item.feedback_user,
             "embedding": item.embedding} for item in comp_data]

# Route pour obtenir les données de `ImportSegmentContxt`
@app.get("/data/contxt")
def get_table_contxt():
    # Récupération des données de la table ImportSegmentContxt
    contxt_data = db_session.query(ImportSegmentContxt).all()
    if not contxt_data:
        raise HTTPException(status_code=404, detail="Table contxt vide")
    
    return [{"id": item.id,
             "ref_user": item.ref_user,
             "segment": item.segment,
             "prediction_contxt": item.prediction_contxt,
             "feedback_user": item.feedback_user,
             "embedding": item.embedding} for item in contxt_data]
