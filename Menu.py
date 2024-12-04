# Importations nécessaires
import pandas as pd  # Manipulation des datasets
import numpy as np  # Calculs mathématiques
import matplotlib.pyplot as plt  # Création de graphiques basiques
import seaborn as sns  # Visualisation avancée
import scipy.stats as stat  # Calculs statistiques
from sklearn.linear_model import LogisticRegression  # Modèle de régression logistique
from sklearn.impute import SimpleImputer  # Gestion des valeurs manquantes
from sklearn.compose import ColumnTransformer  # Transformation des colonnes par type
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Normalisation et encodage
from sklearn.decomposition import PCA  # Réduction de la dimensionnalité
from imblearn.pipeline import Pipeline  # Construction de pipeline pour le traitement des données
from imblearn.over_sampling import SMOTE  # Technique de sur-échantillonnage
from imblearn.under_sampling import RandomUnderSampler  # Technique de sous-échantillonnage
import joblib as jb  # Sauvegarde et chargement des modèles
import streamlit as st  # Interface utilisateur interactive
import plotly.express as px  # Graphiques interactifs
import os  # Gestion des fichiers et chemins
import warnings  # Gestion des avertissements

# Configuration générale
st.set_page_config(
    page_title="Analyse du Marché",
    page_icon=":bar_chart:",
    layout="wide"
)

# Suppression des avertissements inutiles
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Ajout de styles CSS personnalisés
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #808080; /* Couleur de la barre latérale */
    }
    </style>
    """, unsafe_allow_html=True)




# Interface du API
st.header("🎉 Bienvenue dans l'Application de Scoring et Analyse de Prêts 🏦")

st.subheader("Description :")
st.markdown("Cette application vous permet d'explorer vos données, de visualiser des tendances, et de prédire les défauts de paiement grâce à deux modèles de machine learning performants :")
st.subheader("Arbre de décision 🌳 :")
st.markdown("Offre des prédictions transparentes et faciles à comprendre. Utile pour explorer les principaux facteurs influençant le scoring.")
st.subheader("Régression logistique 📈 :")
st.markdown("Fourni des prédictions basées sur des modèles statistiques robustes. Particulièrement adapté pour interpréter la probabilité de défaut.")
st.subheader("Fonctionnalités :")
st.markdown("""
📊 Tableau de bord interactif : Analyse des tendances par âge, revenus, intentions de prêts, etc.
            
🔮 Prédictions personnalisées : Entrez les caractéristiques du client et obtenez des scores détaillés.
            
📁 Gestion flexible des données : Téléchargez vos propres fichiers ou utilisez le dataset intégré.
            
            """)
st.subheader("Instructions :")
st.markdown("""
a) Téléchargez vos données clients (formats pris en charge : .csv, .xlsx).       
b) Explorez les graphiques générés automatiquement.            
c) Fournissez les caractéristiques du client pour obtenir une prédiction en temps réel.
            
Note importante : Les prédictions fournies sont indicatives et doivent être utilisées avec discernement.
""")