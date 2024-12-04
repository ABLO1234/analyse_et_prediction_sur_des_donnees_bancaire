import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import streamlit as st

# Ajout de styles CSS personnalisés
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #808080; /* Couleur de la barre latérale */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Prédiction des modèles")


# Chargement des modèles
try:
    pipe_tree = joblib.load(r"C:\Users\X1 Carbon\Desktop\ABLO'S FOLDER\ONLINE COURSE-SQL-ABLO\PYTHON\PYTHON\MACHINE LEARNING\PROJET DE MACHINE LEARNING\DEFAULT_OF_PAYMENT_MODEL\DATA\ML_MODEL\models\decision_tree.pkl")
except FileNotFoundError:
    st.error("Le modèle 'decision_tree.pkl' est introuvable. Assurez-vous qu'il est dans le dossier 'models'.")
    st.stop()

try:
    pipe_log = joblib.load(r"C:\Users\X1 Carbon\Desktop\ABLO'S FOLDER\ONLINE COURSE-SQL-ABLO\PYTHON\PYTHON\MACHINE LEARNING\PROJET DE MACHINE LEARNING\DEFAULT_OF_PAYMENT_MODEL\DATA\ML_MODEL\models\logistic.pkl")
except FileNotFoundError:
    st.error("Le modèle 'logistic.pkl' est introuvable. Assurez-vous qu'il est dans le dossier 'models'.")
    st.stop()



# Création d'une fonction de prédiction
def model_predic(modele, data_entry):
    try:
        prediction = modele.predict(data_entry)
        prob = modele.predict_proba(data_entry)
        prob = np.round(prob * 100, 3)
        return prediction, prob
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None

# Entrée des caractéristiques utilisateur
st.sidebar.header("LES CARACTÉRISTIQUES DU CLIENT")
age = st.sidebar.number_input("Âge", min_value=0, max_value=100, step=1)
rev_annuel = st.sidebar.number_input("Revenu Annuel ($)", min_value=0.0, step=1000.0)
propriete = st.sidebar.selectbox("Propriété", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
duree_emploi = st.sidebar.number_input("Durée d'Emploi (années)", min_value=0, max_value=50, step=1)
intention = st.sidebar.selectbox(
    "Intention pour laquelle le prêt va être contracté", 
    options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
)
categorie_pret = st.sidebar.selectbox("Catégorie de Prêt", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
montant = st.sidebar.number_input("Montant du Prêt ($)", min_value=0.0, step=100.0)
interet = st.sidebar.number_input("Taux d'Intérêt (%)", min_value=0.0, max_value=100.0, step=0.1)
pourcentage_rev = st.sidebar.number_input("Pourcentage du Revenu (%)", min_value=0.0, max_value=100.0, step=0.1)
defaut_historique = st.sidebar.selectbox("Défaut Historique (Y : Yes, N : No)", options=['Y', 'N'])
duree_credit = st.sidebar.number_input("Durée du Crédit (années)", min_value=0, max_value=30, step=1)

# Création d'un DataFrame avec les caractéristiques saisies
input_data = pd.DataFrame({
    'age': [age],
    'rev_annuel': [rev_annuel],
    'propriete': [propriete],
    'duree_emploi': [duree_emploi],
    'intention': [intention],
    'categorie_pret': [categorie_pret],
    'montant': [montant],
    'interet': [interet],
    'pourcentage_rev': [pourcentage_rev],
    'defaut_historique': [defaut_historique],
    'duree_credit': [duree_credit]
})

# Création d'un bouton de prédiction
if st.sidebar.button("Prédire"):
    pred0, prob0 = model_predic(pipe_tree, input_data)
    pred1, prob1 = model_predic(pipe_log, input_data)

    if prob0 is not None:
        st.subheader("Modèle d'Arbre de Décision")
        df_pred0 = pd.DataFrame({
            'Catégorie0': ["Défaut", "Non Défaut"],
            'Probabilité0': prob0[0]
        })

        # Graphe pour le modèle d'arbre de décision
        bar0 = px.bar(
            df_pred0, 
            y='Probabilité0', 
            color="Catégorie0", 
            color_discrete_sequence=["red", "blue"], 
            text="Probabilité0"
        )
        bar0.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(bar0)

    if prob1 is not None:
        st.subheader("Modèle Logistique")
        df_pred1 = pd.DataFrame({
            'Catégorie1': ["Défaut", "Non Défaut"],
            'Probabilité1': prob1[0]
        })
        
        # Graphe pour le modèle logistique
        bar1 = px.bar(
            df_pred1, 
            y="Probabilité1", 
            color='Catégorie1', 
            text="Probabilité1", 
            color_discrete_sequence=["red", "green"]
        )
        bar1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(bar1)
