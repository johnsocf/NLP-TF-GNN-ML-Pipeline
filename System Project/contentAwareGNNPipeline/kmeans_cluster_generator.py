import nltk
from nltk.cluster import KMeansClusterer

"""
  Uses KMeans to generate clusters from the text entities
"""


class KMeansClusterGenerator():
    def __init__(self, outer_self, data_df, num_clusters=4):
        self.num_clusters = num_clusters
        self.data_df = data_df

    def generate(self):
        return self._find_clusters()

    def _find_clusters(self):
        text_token_segments = [text_preprocessed for text_preprocessed in self.data_df['embedding']]
        clusterer = KMeansClusterer(self.num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
        return clusterer.cluster(text_token_segments, assign_clusters=True)
