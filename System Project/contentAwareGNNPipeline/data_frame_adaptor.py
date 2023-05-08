import pandas as pd
from sklearn.model_selection import train_test_split

"""
  Manages the dataframe mutations needed for the GNN
"""


class DataFrameAdaptor():
    def __init__(self, outer_self):
        self.node_df = outer_self.node_df
        self.edge_df = outer_self.edge_df

        node_train, node_test = self._get_node_train_test_split()
        self.node_train = node_train
        self.node_test = node_test

        edge_train, edge_test = self._get_edge_train_test_split()
        self.edge_train = edge_train
        self.edge_test = edge_test

        node_full_adj, edge_full_adj = self._create_adj_id(self.node_df,
                                                           self._generate_bidirectional_matrices(self.edge_df))
        self.node_full_adj = node_full_adj
        self.edge_full_adj = edge_full_adj

        node_train_adj, edge_train_adj = self._create_adj_id(self.node_train,
                                                             self._generate_bidirectional_matrices(self.edge_train))
        self.edge_train_adj = edge_train_adj
        self.node_train_adj = node_train_adj

    def _get_node_train_test_split(self):
        return train_test_split(self.node_df, test_size=0.15, random_state=42)

    def _get_edge_train_test_split(self):
        edge_train = self.edge_df.loc[~((self.edge_df['source'].isin(self.node_test.index)) | (
            self.edge_df['target'].isin(self.node_test.index)))]
        edge_test = self.edge_df.loc[(self.edge_df['source'].isin(self.node_test.index)) | (
            self.edge_df['target'].isin(self.node_test.index))]
        return edge_train, edge_test

    @staticmethod
    def _generate_bidirectional_matrices(directional_df):
        reverse_df = directional_df.rename(columns={'source': 'target', 'target': 'source'})
        reverse_df = reverse_df[directional_df.columns]
        reverse_df = pd.concat([directional_df, reverse_df], ignore_index=True, axis=0)
        return reverse_df

    @staticmethod
    def _create_adj_id(node_df, edge_df):
        node_df = node_df.reset_index()
        edge_df = pd.merge(edge_df, node_df[['text_name', 'index']].rename(columns={"index": "source_id"}),
                           how='left', left_on='source', right_on='text_name').drop(columns=['text_name'])
        edge_df = pd.merge(edge_df, node_df[['text_name', 'index']].rename(columns={"index": "target_id"}),
                           how='left', left_on='target', right_on='text_name').drop(columns=['text_name'])

        edge_df.dropna(inplace=True)
        edge_df = edge_df.astype({col: 'int32' for col in edge_df.select_dtypes('int64').columns})
        return node_df, edge_df
