import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

class DR:
    def __init__(self, df, target, scale=True):
        """
        Inicializa la clase para reducción de dimensionalidad.

        Parámetros:
        - df (pd.DataFrame): El dataframe con los datos.
        - target (str): Nombre de la variable objetivo.
        - method (str): 'pca', 'tsne' o 'umap'.
        - scale (bool): Si True, aplica escalado a los datos.
        """
        self.df = df.copy()
        self.target = target
        self.scale = scale
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]
        
        if self.scale:
            self.scaler = StandardScaler()
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

    def reduce_and_plot(self,method='pca'):
        """
        Aplica la reducción de dimensionalidad y genera un gráfico scatter.
        """
        self.method = method.lower()
        if self.method == 'pca':
            reducer = PCA(n_components=2)
        elif self.method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif self.method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Método no válido. Usa 'pca', 'tsne' o 'umap'.")

        reduced = reducer.fit_transform(self.X)
        reduced_df = pd.DataFrame(reduced, columns=['Dim1', 'Dim2'])
        reduced_df[self.target] = self.y.values

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=reduced_df, x='Dim1', y='Dim2', hue=self.target, palette='tab10', alpha=0.7)
        plt.title(f'Reducción con {self.method.upper()}')
        plt.tight_layout()
        plt.show()
