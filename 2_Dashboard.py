import pandas as pd  # Manipulation de données
import matplotlib.pyplot as plt  # Graphiques de base
import seaborn as sns  # Graphiques avancés
import streamlit as st  # Interface utilisateur
import plotly.express as px  # Graphiques interactifs
import os  # Gestion des fichiers
import warnings  # Suppression des avertissements inutiles

warnings.filterwarnings('ignore')

# Ajout de styles CSS personnalisés
st.markdown("""
    <style>
    /* Couleur de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #808080; /* Gris */
    }
    /* Couleur des titres principaux */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #800080; /* Violet */
        font-family: 'Arial Black', sans-serif;
    }
    /* Couleur des sous-titres */
    .stMarkdown h4, .stMarkdown h5 {
        color: #FFD700; /* Doré */
    }
    /* Couleur des liens */
    a {
        color: #008080; /* Teal */
    }
    /* Couleur de fond globale */
    body {
        background-color: #F5FFFA; /* Mint Cream */
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.title("🎨 Tableau de bord interactif")
st.write("Bienvenue dans un tableau de bord modernisé avec des couleurs vives et une interface interactive.")


# Fonction pour charger les données
@st.cache_data
def load_data(uploaded_file=None, default_path=None, default_file=None, delimiter=","):
    """
    Charge les données à partir d'un fichier téléchargé ou d'un chemin par défaut.

    Args:
        uploaded_file: Fichier téléchargé par l'utilisateur (Streamlit file uploader).
        default_path (str): Chemin par défaut du répertoire contenant le fichier.
        default_file (str): Nom du fichier par défaut.
        delimiter (str): Délimiteur utilisé dans les fichiers (par défaut : ",").

    Returns:
        pd.DataFrame: Le DataFrame contenant les données.

    Raises:
        ValueError: Si aucune donnée valide ne peut être chargée.
    """

    # Chargement du dataset
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            st.write(f"Fichier chargé : {uploaded_file.name}")
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            raise ValueError("Erreur de chargement des données depuis le fichier téléchargé.")
    elif default_path and default_file:
        try:
            default_full_path = os.path.join(default_path, default_file)
            df = pd.read_csv(default_full_path, encoding="ISO-8859-1", delimiter=delimiter)
            st.warning(f"Aucun fichier n'a été chargé. Utilisation du fichier par défaut : {default_file}")
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier par défaut : {e}")
            raise ValueError("Erreur de chargement des données depuis le fichier par défaut.")
    else:
        st.error("Aucune source de données n'a été spécifiée.")
        raise ValueError("Aucune donnée valide n'a été trouvée.")

# Paramètres pour les chemins
default_path = r"C:\Users\X1 Carbon\Desktop\ABLO'S FOLDER\ONLINE COURSE-SQL-ABLO\PYTHON\PYTHON\MACHINE LEARNING\PROJET DE MACHINE LEARNING\DEFAULT_OF_PAYMENT_MODEL\DATA\ML_MODEL\dataset_used"
default_file = "credit_risk_dataset.csv"

# Fichier chargé par l'utilisateur
fl = st.file_uploader("Téléchargez un fichier :", type=["csv", "txt"])

# Chargement des données
try:
    df = load_data(uploaded_file=fl, default_path=default_path, default_file=default_file, delimiter=";")
except ValueError:
    st.stop()

# Vérification des colonnes nécessaires
required_columns = [
    'cb_person_cred_hist_length', 'person_home_ownership',
    'loan_grade', 'loan_amnt', 'person_emp_length', 'loan_intent'
]
missing_columns = [col for col in required_columns if col not in df.columns]

# Gestion des erreurs
if missing_columns:
    st.error(f"Les colonnes suivantes sont manquantes : {missing_columns}")
    st.stop()


# Répartition des âge
st.subheader("🎂 Évolution de l'âge des clients")
age = px.histogram(data_frame=df, x="person_age",labels={"person_age" : "Age des clients", "count" : "Clients"})

st.plotly_chart(age)

# Répartition des revenus
st.subheader("💰 Évolution du revenu annuel des clients")
revenu = px.histogram(data_frame=df, x = "person_income", color_discrete_sequence=["#F5FFFA"])
st.plotly_chart(revenu)

# Section : Évolution de l'historique de crédit
st.subheader("📊 Évolution de l'historique de crédit")
fig0 = px.line(
    df, 
    y='cb_person_cred_hist_length', 
    x=df.index, 
    color_discrete_sequence=["#ADD8E6"]  # Bleu clair
)
st.plotly_chart(fig0)

# Section : Répartition des clients selon leur logement
st.subheader("🏠 Répartition des clients selon leur logement")
pie_df = df['person_home_ownership'].value_counts().reset_index()
pie_df.columns = ['person_home_ownership', 'count']
fig1 = px.pie(
    pie_df, 
    names='person_home_ownership', 
    values='count', 
    color_discrete_sequence=["#90EE90", "#FFA500", "#FF5733", "#800080"]  # Couleurs
)
st.plotly_chart(fig1)

# Section : Répartition des prêts par catégorie
st.subheader("💼 Répartition des prêts par catégorie")
loan_grade_df = df['loan_grade'].value_counts().reset_index()
loan_grade_df.columns = ['loan_grade', 'count']
fig2 = px.pie(
    loan_grade_df,
    names='loan_grade', 
    values='count',
    color_discrete_sequence=["#FFD700", "#FFB6C1", "#008080"]  # Couleurs
)
st.plotly_chart(fig2)

# Section : Montant moyen du prêt par propriété
st.subheader("💰 Montant moyen des prêts par propriété")
line_chart = df.groupby(by="person_home_ownership")["loan_amnt"].mean().reset_index()
fig3 = px.bar(
    line_chart, 
    x="person_home_ownership", 
    y="loan_amnt", 
    text="loan_amnt", 
    color_discrete_sequence=["#8B0000"],  # Rouge foncé
    labels={"person_home_ownership": "Propriété", "loan_amnt": "Montant moyen ($)"}
)
fig3.update_traces(textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

# Section : Corrélation entre durée d'emploi et montant du prêt
st.subheader("📈 Corrélation : Durée d'emploi vs Montant du prêt")
fig4 = px.scatter(
    df, 
    x="person_emp_length", 
    y="loan_amnt", 
    color='loan_intent', 
    labels={"person_emp_length": "Durée d'emploi (années)", "loan_amnt": "Montant du prêt ($)"}
)
st.plotly_chart(fig4)


