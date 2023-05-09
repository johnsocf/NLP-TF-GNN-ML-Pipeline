# NLP-TF-GNN-ML-Pipeline

# Text Preprocessing

## Description
ContentAwareGNNPipeline/main.py runs a graph building pipeline using ML processes for Clustering and Topic Modeling as subprocesses.  
It can be run from the ContentAwareGNNPipeline directory by the command 'python3 main.py'.  The results are difficult to interpret via this approach although the validation metrics will print, as will markers which denote the milestones the pipeline is reaching as it runs.

Since this is an experimental pipeline it is best run through the python notebook, Cat_Johnson_EN.705.603.82_Systems_Project_2023.ipynb so that the introduction, analysis, data collections from the processes, and explanations can be read along with the code.



## To Run from python script
run
python3 ContentAwareGNNPipeline/main.py

## The Processes Performed:
1. Generates topics from each text sample using Linear Discriminant Analysis (LDA)
2. Generates clusters from the text using K-Means Clustering
3. Preprocesses data into a node table and an edge table for building the GNN tensors and deep learning layers.  This defines the graph based on the text nodes, attributes and text to text relationships.  
4. Adapts the data frames for the graph so that nodes are associated with indices and graph edges point to those indices based on the nodes they connect.  Generates data sets for training and testing the GNN.
5. Builds the graph tensors and Keras layers for the Neural Network
6. Compiles, Fits, and Predicts results from the pipeline.


## The Modules:
  CategoryAwareTFGNN: Runs the pipeline processes.

  DocumentTopicGenerator: Generates topics from each text sample using 
  Linear Discriminant Analysis (LDA) via a Gensim LDA Multicore Model
  Exposes the update_df method to add Topics to the dataframe from the 'text'
  column.

  KMeansClusterGenerator: Generates clusters from the text.
  -- The num_clusters parameter will set the number of clusters to generate.

  DataPreProcessor: Preprocesses the data
  -- Builds the node dataframe which contains:
    -- node name
    -- each of it's attributes
  -- Builds the edge dataframe which contains:
    -- node source
    -- node target
    -- each edge type

  DataFrameAdaptor: Adapting Data For the Graph.
  -- Defines nodes by index
  -- Updates graph edge node references to point to ids by index
  -- Generates training and test sets
  -- Generates a full adjacency matrix for testing

  TFGraph: Builds the graph structure and deep learning layers
  -- Builds the graph structure using tensors
  -- Builds the deep learning layers using Keras

  TFGNN: Exposes API methods to work with the model:
  -- get_model
  -- compile
  -- fit (train)
  -- predict

  PipelineValidation: Exposes API methods for validation:
  -- validate_clustering: Validates clusters against category labels using the 
      edges to see if the categorization is shared between nodes.  Returns the 
      accuracy of whether the category and cluster are both shared between 
      nodes.
  -- validate_topic_clusters: Validates primary topic cluster embeddings to see 
      how their dimensions deviate from the centroid of the cluster and the 
      neighboring clusters.  This gives context for whether key topics in a 
      cluster align with other topics in the same dimensional space.
  -- The CategoryAwareTFGNN pipeline's 'process' function returns the predictions of the pipeline.

