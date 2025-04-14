import streamlit as st
import os
import runpy

st.set_page_config(page_title="TP2 Clustering", layout="wide")

st.title("🚀 Redirection en cours...")
st.markdown("Vous allez être redirigé automatiquement vers la page principale...")

# Appelle le script principal manuellement
runpy.run_path("pages/singbo_davy.py")
