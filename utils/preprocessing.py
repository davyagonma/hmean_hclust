import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_csv(file):
    df = pd.read_csv(file)
    df = df.select_dtypes(include=[np.number])  # On garde les colonnes numériques
    df = df.dropna()
    return df


def manual_entry(raw_text):
    """
    Prend une chaîne de type :
    1.2, 3.4, 5.6
    7.8, 9.0, 2.3
    """
    try:
        lines = raw_text.strip().split("\n")
        data = [list(map(float, line.strip().split(","))) for line in lines]
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()


def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df.values)
    return pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
