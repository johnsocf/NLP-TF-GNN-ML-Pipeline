import pandas as pd
import numpy as np
from math import sqrt

"""
  Manages the preprocessing to collect and generate nodes and edges
"""


class DataPreProcessor():
    def __init__(self, outer_self):

        self.data_df = outer_self.data_df
        self.labels = outer_self.df_category_labels
        self.df_category_labels = outer_self.df_category_labels
        self.index_by_topic_map = outer_self.index_by_topic

    def process(self):
        nodes, edges = self._process_training_data_to_graph_entities()
        return nodes, edges

    """
      Builds the node and edge dataframes from the data.
      The node dataframe:
        -- node name
        -- each of it's attributes
          -- topics
      The edge dataframe:
        -- node source
        -- node target
        -- each edge type
      @return: node dataframe, edge dataframe
    """

    def _process_training_data_to_graph_entities(self):
        # Set up matrix variables
        node_cols = ["text_name", "category_num", 'cluster', "text_topic_1_index", "text_topic_2_index",
                     "text_topic_3_index"]
        # category_node_cols = ["category_name", "category_num"]
        graph_df_nodes = pd.DataFrame(columns=node_cols)
        edge_cols = ["source", "target", "shares_category", "shares_cluster", "cos_sim_target_with_source"]
        graph_df_edges = pd.DataFrame(columns=edge_cols)
        text_index = 0

        for text_index_i in range(0, len(self.data_df)):

            text_embedding_i = self.data_df['embedding'][text_index_i]

            category_name_i = self.labels[self.data_df['label'][text_index_i]]
            text_category_i = self.df_category_labels[self.data_df['label'][text_index_i]]
            label_i = self.data_df['label'][text_index_i]
            cluster_i = self.data_df['kmeans_gen_cluster'][text_index_i]

            graph_df_nodes.loc[len(graph_df_nodes.index)] = [
                # text name
                "text-" + str(text_index_i),
                # category num
                self.data_df['label'][text_index_i].astype(np.int32),
                # cluster
                cluster_i.astype(np.int32),
                # text topic 1 index
                np.int32(self.index_by_topic_map[self.data_df['text_topic_1'][text_index_i]]),
                # text topic 2 index
                np.int32(self.index_by_topic_map[self.data_df['text_topic_2'][text_index_i]]),
                # text topic 3 index
                np.int32(self.index_by_topic_map[self.data_df['text_topic_3'][text_index_i]])
            ]
            for text_index_j in range(0, len(self.data_df)):
                text_embedding_j = self.data_df['embedding'][text_index_j]

                text_category_j = self.df_category_labels[self.data_df['label'][text_index_j]]
                label_j = self.data_df['label'][text_index_j]
                cluster_j = self.data_df['kmeans_gen_cluster'][text_index_j]

                shares_category_binary = 0
                if label_i == label_j:
                    shares_category_binary = 1

                shares_cluster_binary = 0
                if cluster_i == cluster_j:
                    shares_cluster_binary = 1

                graph_df_edges.loc[len(graph_df_edges.index)] = [
                    # source
                    "text-" + str(text_index_i),
                    # target
                    "text-" + str(text_index_j),
                    # shares category
                    np.int32(shares_category_binary),
                    # shares_cluster
                    np.int32(shares_cluster_binary),
                    # cos_sim_target_with_source,
                    self._cos_similarity(text_embedding_i, text_embedding_j)
                ]
        return graph_df_nodes, graph_df_edges

    @staticmethod
    def _squared_sum(x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def _cos_similarity(self, x, y):
        return round(sum(a * b for a, b in zip(x, y)) / float(self._squared_sum(x) * self._squared_sum(y)), 3)