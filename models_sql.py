# models.py

from sqlalchemy import Column, String, Integer, Float, ARRAY, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

from werkzeug.security import generate_password_hash, check_password_hash

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta


# Déclarer la base pour les modèles SQLAlchemy
Base = declarative_base()

#DATABASE_URL = "postgresql://postgres:postgre@localhost/db_pco"
#engine = create_engine(DATABASE_URL)

# Assurer que la table existe (sans conflit)
#Base.metadata.create_all(engine)

# MODELE TABLE OFFRES_EXTRACT
class OffresExtract(Base):
    __tablename__ = 'offres_extract'

    id = Column(Integer, primary_key=True)
    id_offre = Column(String, primary_key=True)
    intitule = Column(String, nullable=False)
    description = Column(Text, nullable=True)

# MODELES DES TABLES DE MONITORING
class ImportSegmentContxt(Base):
    __tablename__ = 'table_monitoring_contxt'

    id = Column(Integer, primary_key=True)
    ref_user = Column(String)
    segment = Column(String)
    prediction_contxt = Column(Integer)
    feedback_user = Column(String)
    embedding = Column(ARRAY(Float))

class ImportSegmentComp(Base):
    __tablename__ = 'table_monitoring_comp'

    id = Column(Integer, primary_key=True)
    ref_user = Column(String)
    segment = Column(String)
    prediction = Column(Integer)
    feedback_user = Column(String)
    embedding = Column(ARRAY(Float))


# MODELE TABLE USERS + AUTHENTIFICATION API_DATA
# Clé secrète et algo de signature 
SECRET_KEY = "1234"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Durée de validité du token

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    ACCESS_TOKEN_EXPIRE_MINUTES = 30 
    # Création du token JWT
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt