from tf_gnn_generator import TFGNN
from tf_graph import TFGraph
from data_preprocessor import DataPreProcessor
from kmeans_cluster_generator import KMeansClusterGenerator
from topic_generator import DocumentTopicGenerator
from data_frame_adaptor import DataFrameAdaptor
from pipeline_validation import PipelineValidation

import numpy as np
import re
from flair.data import Sentence
from flair.embeddings import BertEmbeddings

"""
  Contains Modular Classes for:

  DocumentTopicGenerator:

  KMeansClusterGenerator:

  DataPreProcessor: Preprocesses the data
  -- Builds the node dataframe which contains:
    -- node name
    -- each of it's attributes
  -- Builds the edge dataframe which contains:
    -- node source
    -- node target
    -- each edge type

  DataFrameAdaptor: Adapting Data For the Graph
  -- Defines nodes by index
  -- Updates graph edge node references to point to ids by index
  -- Generates training and test sets
  -- Generates a full adjacency matrix for testing

  TFGraph: Builds the graph structure and deep learning layers
  -- Builds the graph structure using tensors
  -- Builds the deep learning layers using Keras

  TFGNN: Exposes API methods to work with the model:
  -- _get_model
  -- _compile
  -- _fit (train)
  -- _predict

  @param training_data_df: pandas dataframe
  @param training_data_df: pandas dataframe
  @param is_node_prediction: boolean
  @return: node dataframe, edge dataframe
"""


class CategoryAwareTFGNN():
    def __init__(self, training_data_df, labels, is_node_prediction=True):
        self.bert_embedding = BertEmbeddings()
        data_df = training_data_df
        data_df['text_preprocessed'] = self._clean_text(data_df['text'])
        data_df['embedding'] = [self.tokenize(token) for token in data_df['text_preprocessed']]
        data_df = DocumentTopicGenerator(data_df=data_df).update_df()
        print('topics generated')
        data_df['kmeans_gen_cluster'] = KMeansClusterGenerator(self, data_df=data_df).generate()
        print('clusters generated')
        self.data_df = data_df
        index_by_topic, topic_by_index = self._get_index_topic_maps()
        self.index_by_topic = index_by_topic
        self.topic_by_index = topic_by_index

        self.df_category_labels = labels

        # Preprocess data into nodes and edges
        data_pre_processor = DataPreProcessor(outer_self=self)
        print('data preprocessed')
        nodes, edges = data_pre_processor.process()

        # Adapt dataframes for graph
        self.node_df = nodes
        self.edge_df = edges
        self.dataFrameAdaptor = DataFrameAdaptor(outer_self=self)
        print('data adapted')

        # Build TF-Graph
        self.tfGraph = TFGraph(outer_self=self)

        # Set up datasets.  Edge predictions can be requested by @param is_node_prediction
        training_dataset = self.tfGraph.train_node_dataset
        validation_dataset = self.tfGraph.full_node_dataset
        prediction_dataset = self.tfGraph.full_node_dataset
        if not is_node_prediction:
            training_dataset = self.tfGraph.train_edge_dataset
            validation_dataset = self.tfGraph.full_edge_dataset
            prediction_dataset = self.tfGraph.full_edge_dataset

        # Build the graph tensors and keras layers
        self.tfGNN = TFGNN(outer_self=self,
                           num_graph_updates=3,
                           training_dataset=training_dataset,
                           validation_dataset=validation_dataset,
                           prediction_dataset=prediction_dataset)
        print('tfgnn built')

    """
      Compiles, trains, and generates predictions from the model.
      This can be called after instantiating the CategoryAwareTFGNN for the pipeline results.
      @return: predictions from the model generated by the pipeline
    """
    def process(self):
        self.tfGNN.compile()
        self.tfGNN.fit()
        return self.tfGNN.predict()

    @staticmethod
    def _clean_text(column):
        df_column_without_punctuation = column.map(lambda x: re.sub('[,\.!?]', '', x))
        return df_column_without_punctuation.map(lambda x: x.lower())

    def tokenize(self, token):
        word = Sentence(token)
        self.bert_embedding.embed(word)
        return word[0].embedding.numpy()

    def _get_index_topic_maps(self):
        dataset_length = len(self.data_df.index)
        topics = self.data_df['text_topic_1'].values.tolist() + self.data_df['text_topic_2'].values.tolist() + \
                 self.data_df['text_topic_3'].values.tolist()
        topics = sum(
            [[self.data_df['text_topic_1'][i], self.data_df['text_topic_2'][i], self.data_df['text_topic_3'][i]] for i
             in range(0, dataset_length)], [])
        index_by_topic = {}
        for indx, topic in enumerate(topics):
            if topic not in index_by_topic:
                index_by_topic[topic] = indx
        topic_by_index = {v: k for k, v in index_by_topic.items()}
        return index_by_topic, topic_by_index

    """
    @return: The % accuracy of whether the category and cluster are both
    shared between nodes.
    """

    def validate_clustering(self):
        return PipelineValidation(outer_self=self).validate_clustering()

    """
    @return: A map with the average amount of topic embedding deviation from the
    centroid of each cluster for the embeddings from the requested cluster. 
    The requested cluster is marked by the key name 'cluster_self' instead of a 
    cluster number in the map.
    """

    def validate_topic_clusters(self, cluster=0):
        return PipelineValidation(
            outer_self=self,
            is_topic_cluster_validation=True)\
            .validate_topic_clusters(cluster_requested=cluster)
