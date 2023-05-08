import tensorflow_gnn as tfgnn
import tensorflow as tf

"""
  Exposes API methods to:
  -- Build the model
  -- Compile the model
  -- Fit the model (training)
  -- Predicting from the model

  These methods need to be called in order.  However multiple predictions can be made on the same model.
"""


class TFGNN():
    def __init__(self,
                 outer_self,
                 num_graph_updates,
                 training_dataset,
                 validation_dataset,
                 prediction_dataset,
                 steps=10, epochs=100, loss=None, metrics=None):
        # Graph update variables
        self.input_graph = outer_self.tfGraph.input_graph
        self.dense_layer = outer_self.tfGraph.dense_layer
        self.set_initial_node_state = outer_self.tfGraph.set_initial_node_state
        self.set_initial_edge_state = outer_self.tfGraph.set_initial_edge_state

        self.num_graph_updates = num_graph_updates
        self.model = self._get_model()

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.prediction_dataset = prediction_dataset

        # Training params
        self.steps_per_epoch = steps
        self.epochs = epochs

    def _get_model(self):
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=self.set_initial_node_state,
            edge_sets_fn=self.set_initial_edge_state
        )(self.input_graph)

        graph_updates = self.num_graph_updates
        for i in range(graph_updates):
            graph = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    'articles': tfgnn.keras.layers.NodeSetUpdate({
                        'topics': tfgnn.keras.layers.SimpleConv(
                            message_fn=self.dense_layer(32),
                            reduce_type="sum",
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            self.dense_layer(64)))})(graph)  # start here

            logits = tf.keras.layers.Dense(1, activation='softmax')(graph.node_sets["articles"][tfgnn.HIDDEN_STATE])
        return tf.keras.Model(self.input_graph, logits)

    def compile(self):
        self.model.compile(
            tf.keras.optimizers.Adam(learning_rate=0.3),
            # loss = 'categorical_crossentropy',
            # metrics = ['categorical_accuracy']
        )

    def fit(self):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100,
                                              restore_best_weights=True)
        self.model.fit(self.training_dataset.repeat(),
                       validation_data=self.validation_dataset,
                       steps_per_epoch=self.steps_per_epoch,
                       epochs=self.epochs,
                       callbacks=[es])

    def predict(self):
        return self.model.predict(self.prediction_dataset)