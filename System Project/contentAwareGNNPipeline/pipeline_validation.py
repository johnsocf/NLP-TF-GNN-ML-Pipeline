import numpy as np


class PipelineValidation():
    def __init__(self, outer_self, is_topic_cluster_validation=False):
        self.node_df = outer_self.node_df
        self.edge_df = outer_self.edge_df
        self.topic_by_index = outer_self.topic_by_index
        self.tokenize = outer_self.tokenize
        self.index_by_topic = outer_self.index_by_topic
        if is_topic_cluster_validation:
            self.embedding_by_token = {v: self.tokenize(self.topic_by_index[v]) for k, v in self.index_by_topic.items()}
            self.topic_embeddings_by_cluster = self._get_topic_embeddings_by_clusters()
            self.top_topic_embeddings_by_cluster = self._get_top_topic_embeddings_by_clusters()
            self.topic_embedding_mean_by_cluster = self._get_topic_embedding_mean_by_cluster()

    """
      Validates clusters against category labels using the edges to see
      if the categorization is shared between nodes.

      This validating assumes that a category is a dimensional space which can be
      have boundaries represented by the dimensions of text token embeddings.

      @return: The % accuracy of whether the category and cluster are both
      shared between nodes.
    """

    def validate_clustering(self):
        cluster_consistent_with_catgory_count = len(
            self.edge_df[self.edge_df["shares_category"] == self.edge_df["shares_cluster"]])
        return cluster_consistent_with_catgory_count / len(self.edge_df)

    """
      Validates primary topic cluster embeddings to see how their dimensions deviate from
      the centroid of the cluster and the neighboring clusters.  This gives
      context for whether key topics in a cluster align with other topics in the
      same dimensional space.


      This validating assumes that a category is a dimensional space which can be
      have boundaries represented by the dimensions of text token embeddings.

      @param cluster_requested: The cluster embeddings that the result is in context of. 
      @return: The % accuracy of whether the category and cluster are both
      shared between nodes.
    """

    def validate_topic_clusters(self, cluster_requested):
        # map {cluster number: topic embedding deviations from mean for that cluster []}
        topic_embed_deviation_from_mean_for_cluster = {}

        # cluster in context of validation.  topic 1 embeddings only
        cluster_embedding_list = self.top_topic_embeddings_by_cluster[cluster_requested]

        # loop over each cluster's topic embedding mean
        for cluster, embedding_list in self.topic_embedding_mean_by_cluster.items():
            embedding_mean_for_cluster = self.topic_embedding_mean_by_cluster[cluster]
            difference_curr = np.zeros(len(embedding_mean_for_cluster), dtype='float32')
            # loop over the requested cluster's topic embeddings
            for token_embedding in cluster_embedding_list:
                difference = np.absolute(np.subtract(token_embedding, embedding_mean_for_cluster))
                difference_curr = np.add(difference_curr, difference)

            # Get the average difference for each embedding and sum them up for this cluster
            map_key = cluster
            if cluster == cluster_requested:
                map_key = 'cluster_self'
            topic_embed_deviation_from_mean_for_cluster[map_key] = np.sum(
                np.divide(difference_curr, len(cluster_embedding_list)))
        return topic_embed_deviation_from_mean_for_cluster

    def _get_topic_embeddings_by_clusters(self):
        # map {cluster number: topic embeddings []}
        topic_embed_by_cluster = {}
        df = self.node_df.filter(
            items=['cluster', "text_topic_1_index", "text_topic_2_index", "text_topic_3_index"])
        for i, row in enumerate(df.itertuples(), 1):
            cluster = row.cluster
            topic_embeddings = [
                self.embedding_by_token[row.text_topic_1_index],
                self.embedding_by_token[row.text_topic_2_index],
                self.embedding_by_token[row.text_topic_3_index]]
            if cluster in topic_embed_by_cluster:
                topic_embed_by_cluster[cluster] = topic_embed_by_cluster[cluster] + topic_embeddings
            else:
                topic_embed_by_cluster[cluster] = topic_embeddings
        return topic_embed_by_cluster

    def _get_top_topic_embeddings_by_clusters(self):
        # map {cluster number: topic embeddings []}
        top_topic_embed_by_cluster = {}
        df = self.node_df.filter(
            items=['cluster', "text_topic_1_index", "text_topic_2_index", "text_topic_3_index"])
        for i, row in enumerate(df.itertuples(), 1):
            cluster = row.cluster
            topic_embedding = [self.embedding_by_token[row.text_topic_1_index]]
            if cluster in top_topic_embed_by_cluster:
                top_topic_embed_by_cluster[cluster] = top_topic_embed_by_cluster[cluster] + topic_embedding
            else:
                top_topic_embed_by_cluster[cluster] = topic_embedding
        return top_topic_embed_by_cluster

    def _get_topic_embedding_mean_by_cluster(self):
        # map {cluster number: topic embedding mean []}
        topic_embedding_mean_by_cluster = {}
        for cluster, embedding_list in self.topic_embeddings_by_cluster.items():
            topic_embedding_mean_by_cluster[cluster] = np.average(embedding_list, axis=0)
        return topic_embedding_mean_by_cluster