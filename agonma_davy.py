import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.clustering import kmeans, hclust
from utils.metrics import intra_cluster_distance, inter_cluster_distance, elbow_method
from utils.preprocessing import load_csv, manual_entry, apply_pca

st.set_page_config(page_title="AGONMA Clustering TP2", layout="wide")

st.title("🧠 TP2 - Clustering from Scratch (K-means & Hclust)")
st.markdown("Par **AGONMA Singbo Davy** – Master GL IFRI")

# Sidebar config
st.sidebar.title("🔧 Configuration")

data_input_mode = st.sidebar.radio("Mode de chargement", ["Par fichier CSV", "Manuel"])

# --- 1. Chargement des données ---
df = None
if data_input_mode == "Par fichier CSV":
    file = st.sidebar.file_uploader("Uploader un fichier CSV", type=["csv"])
    if file:
        df = load_csv(file)
else:
    raw_data = st.sidebar.text_area("Entrez vos données (valeurs séparées par virgule)", height=200,
                                     placeholder="1.2, 3.4, 5.6\n7.8, 9.0, 2.3")
    if raw_data:
        df = manual_entry(raw_data)

if df is not None and not df.empty:
    st.subheader("📊 Aperçu des données")
    st.dataframe(df.head())

    # --- 2. Suggestion de K ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Analyse pour le choix de K")
    show_elbow = st.sidebar.checkbox("Afficher la méthode du coude")
    k = st.sidebar.number_input("Nombre de clusters (K)", min_value=2, max_value=10, value=3, step=1)

    if show_elbow:
        k_range = list(range(2, 11))
        inertias = elbow_method(df.values, k_range)

        fig_elbow = plt.figure()
        plt.plot(k_range, inertias, marker="o")
        plt.xlabel("K")
        plt.ylabel("Inertie intra-cluster")
        plt.title("Méthode du coude")
        st.pyplot(fig_elbow)

    # --- 3. K-means ---
    st.subheader("🔷 Résultats K-means")
    centroids, labels = kmeans(df.values, k)

    st.write("**Centroïdes :**")
    st.dataframe(pd.DataFrame(centroids, columns=df.columns))

    st.write("**Écart-type par cluster :**")
    std_by_cluster = []
    for i in range(k):
        cluster_points = df.values[labels == i]
        std_by_cluster.append(np.std(cluster_points, axis=0))
    st.dataframe(pd.DataFrame(std_by_cluster, columns=df.columns))

    # --- Visualisation ---
    if df.shape[1] > 2:
        df_visu = apply_pca(df)
    else:
        df_visu = df.copy()

    df_visu["Cluster"] = labels
    fig = plt.figure()
    sns.scatterplot(data=df_visu, x=df_visu.columns[0], y=df_visu.columns[1], hue="Cluster", palette="Set2")
    plt.title("Visualisation des clusters (K-means)")
    st.pyplot(fig)

    # --- Métriques ---
    intra = intra_cluster_distance(df.values, labels, centroids)
    inter = inter_cluster_distance(centroids)

    st.markdown(f"**📈 Distance intra-cluster :** {intra:.3f}")
    st.markdown(f"**📉 Distance inter-cluster :** {inter:.3f}")

    # --- 4. Hclust ---
    st.subheader("🌿 Hierarchical Clustering (Hclust)")
    dendro_btn = st.button("Générer le dendrogramme")

    if dendro_btn:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist

        # On utilise scipy uniquement pour afficher le dendrogramme
        linked = linkage(df.values, 'single')
        fig2 = plt.figure(figsize=(10, 5))
        dendrogram(linked)
        st.pyplot(fig2)

    # --- 5. Prédiction d’un nouveau point ---
    st.sidebar.subheader("🧪 Prédiction")
    new_point = st.sidebar.text_input("Coordonnées du nouveau point (ex: 2.5, 3.6)")
    if new_point:
        try:
            point = np.array(list(map(float, new_point.split(","))))
            distances = [np.linalg.norm(point - c) for c in centroids]
            assigned = int(np.argmin(distances))
            st.sidebar.success(f"Ce point appartient au cluster {assigned}")
        except:
            st.sidebar.error("Format invalide.")

else:
    st.info("Veuillez charger des données pour commencer.")
