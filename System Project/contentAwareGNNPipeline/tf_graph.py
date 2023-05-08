import numpy as np
import tensorflow_gnn as tfgnn
import tensorflow as tf

"""
  Builds the graph structure for the GNN from the node and edge dataframes using tfgnn.GraphTensor, tfgnn.NodeSet, and tfgnn.EdgeSet.
  Builds the deep learning layers for the GNN using Keras and makes these available for fitting the TF-GNN model
"""


class TFGraph():

    def __init__(self, outer_self):
        full_tensor = self._create_graph_tensor(outer_self.dataFrameAdaptor.node_full_adj,
                                                outer_self.dataFrameAdaptor.edge_full_adj)
        train_tensor = self._create_graph_tensor(outer_self.dataFrameAdaptor.node_train_adj,
                                                 outer_self.dataFrameAdaptor.edge_train_adj)

        self.full_node_dataset = self._generate_dataset_from_graph(full_tensor, self._node_batch_merge)
        self.train_node_dataset = self._generate_dataset_from_graph(train_tensor, self._node_batch_merge)

        self.full_edge_dataset = self._generate_dataset_from_graph(full_tensor, self._edge_batch_merge)
        self.train_edge_dataset = self._generate_dataset_from_graph(train_tensor, self._edge_batch_merge)

        self.set_initial_node_state = self._set_initial_node_state
        self.set_initial_edge_state = self._set_initial_edge_state

        graph_spec = self.train_edge_dataset.element_spec[0]
        self.input_graph = tf.keras.layers.Input(type_spec=graph_spec)
        print("graph spec compatibility: ", graph_spec.is_compatible_with(full_tensor))

        self.dense_layer = self._dense_layer

    @staticmethod
    def _create_graph_tensor(node_df, edge_df):
        return tfgnn.GraphTensor.from_pieces(
            node_sets={
                "articles": tfgnn.NodeSet.from_fields(
                    sizes=[len(node_df)],
                    features={
                        'article_category': np.array(node_df['category_num'], dtype='int32').reshape(len(node_df),
                                                                                                     1),
                        'article_cluster': np.array(node_df['cluster'], dtype='int32').reshape(len(node_df), 1),
                        'article_topic_1_id': np.array(node_df['text_topic_1_index'], dtype='int32').reshape(
                            len(node_df), 1),
                        'article_topic_2_id': np.array(node_df['text_topic_2_index'], dtype='int32').reshape(
                            len(node_df), 1),
                        'article_topic_3_id': np.array(node_df['text_topic_3_index'], dtype='int32').reshape(
                            len(node_df), 1)
                    })
            },
            edge_sets={
                "topics": tfgnn.EdgeSet.from_fields(
                    sizes=[len(edge_df)],
                    features={
                        'topics_shared': np.array(edge_df['shares_category'],
                                                  dtype='int32').reshape(len(edge_df), 1),
                        'cosine_similarity': np.array(edge_df['cos_sim_target_with_source'],
                                                      dtype='float32').reshape(len(edge_df), 1)},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("articles", np.array(edge_df['source_id'], dtype='int32')),
                        target=("articles", np.array(edge_df['target_id'], dtype='int32'))))
            }
        )

    @staticmethod
    def _node_batch_merge(graph):
        graph = graph.merge_batch_to_components()
        node_features = graph.node_sets['articles'].get_features_dict()
        edge_features = graph.edge_sets['topics'].get_features_dict()

        label = node_features.pop('article_category')
        _ = edge_features.pop('topics_shared')

        new_graph = graph.replace_features(
            node_sets={'articles': node_features},
            edge_sets={'topics': edge_features})
        return new_graph, label

    @staticmethod
    def _edge_batch_merge(graph):
        graph = graph.merge_batch_to_components()
        node_features = graph.node_sets['articles'].get_features_dict()
        edge_features = graph.edge_sets['topics'].get_features_dict()

        _ = node_features.pop('article_category')
        label = edge_features.pop('topics_shared')

        new_graph = graph.replace_features(
            node_sets={'articles': node_features},
            edge_sets={'topics': edge_features})
        return new_graph, label

    @staticmethod
    def _generate_dataset_from_graph(graph, function):
        dataset = tf.data.Dataset.from_tensors(graph)
        dataset = dataset.batch(32)
        return dataset.map(function)

    @staticmethod
    def _set_initial_node_state(node_set, node_set_name):
        features = [
            tf.keras.layers.Dense(32, activation="relu")(node_set['article_cluster']),
            tf.keras.layers.Dense(32, activation="relu")(node_set['article_topic_1_id']),
            tf.keras.layers.Dense(32, activation="relu")(node_set['article_topic_2_id']),
            tf.keras.layers.Dense(32, activation="relu")(node_set['article_topic_3_id'])
        ]
        return tf.keras.layers.Concatenate()(features)

    @staticmethod
    def _set_initial_edge_state(edge_set, edge_set_name):
        features = [
            tf.keras.layers.Dense(32, activation="relu")(edge_set['cosine_similarity'])
        ]
        return tf.keras.layers.Concatenate()(features)

    @staticmethod
    def _dense_layer(units=64, l2_reg=0.1, dropout=0.25, activation='relu'):
        regularizer = tf.keras.regularizers.l2(l2_reg)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout)])
