from mlflow.tracking import MlflowClient

# Fonction pour enregistrer un modèle
def register_model(model_name, artifact_uri, run_id, client):
    
    try:
        # Vérifier si le modèle existe déjà dans le registre
        model_info = client.get_registered_model(model_name)
    except:
        # Si le modèle n'existe pas, l'enregistrer
        model_info = client.create_registered_model(model_name)

    # Enregistrer une nouvelle version pour le modèle
    model_version = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id
    )
    
    return model_version.version

# Fonction pour promouvoir un modèle en Production
def promote_model(model_name, new_accuracy, previous_accuracy, run_id, artifact_uri):
    """
    Compare les performances et promeut le modèle à l'étape Production s'il est meilleur.
    """
    client = MlflowClient()  # Définir le client dans cette fonction
    
    # Enregistrer le modèle dans le registre
    version = register_model(model_name, artifact_uri, run_id, client)

    if new_accuracy > previous_accuracy:
        # Promouvoir le modèle à l'étape Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        return version  # Retourner la version du modèle qui a été promu
    else:
        return None  # Retourner None si le modèle n'est pas promu

# Informations sur les modèles
run_id = "660506006876858877"

# Modèle rfc_1
artifact_uri_rfc_1 = "mlflow-artifacts:/660506006876858877/887bff6ead2b4df89edc724f042546b0/artifacts/rfc_1"
model_name_rfc_1 = "rfc_1"
new_accuracy_rfc_1 = 0.85
previous_accuracy_rfc_1 = 0.84

# Modèle rfc_2
artifact_uri_rfc_2 = "mlflow-artifacts:/660506006876858877/a022c3a186424fa4af7a7cf3c33c5559/artifacts/rfc_2"
model_name_rfc_2 = "rfc_2"
new_accuracy_rfc_2 = 0.90
previous_accuracy_rfc_2 = 0.89

# Enregistrer et promouvoir les modèles dans le registre
try:
    version_rfc_1 = promote_model(model_name_rfc_1, new_accuracy_rfc_1, previous_accuracy_rfc_1, run_id, artifact_uri_rfc_1)
    if version_rfc_1:
        print(f"RFC_1 registered and promoted to Production with version: {version_rfc_1}")
    else:
        print("RFC_1 not promoted (accuracy lower or equal to previous version)")

except Exception as e:
    print(f"Error registering and promoting model rfc_1: {str(e)}")

try:
    version_rfc_2 = promote_model(model_name_rfc_2, new_accuracy_rfc_2, previous_accuracy_rfc_2, run_id, artifact_uri_rfc_2)
    if version_rfc_2:
        print(f"RFC_2 registered and promoted to Production with version: {version_rfc_2}")
    else:
        print("RFC_2 not promoted (accuracy lower or equal to previous version)")

except Exception as e:
    print(f"Error registering and promoting model rfc_2: {str(e)}")
