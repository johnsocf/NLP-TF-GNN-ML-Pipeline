from category_aware_gnn_pipeline import CategoryAwareTFGNN

import pandas as pd
import datasets as ds


def main():
    training_data_df, labels = collect_training_data(20)
    print('training data collected')
    ca_tfgnn = CategoryAwareTFGNN(training_data_df=training_data_df, labels=labels)
    print('pipeline complete.')
    print('clustering accuracy: ')
    print(ca_tfgnn.validate_clustering())
    print('topic generation deviation from mean of the context cluster and neighboring clusters: ')
    for i in range(0, 4):
        print(ca_tfgnn.validate_topic_clusters(i))
    predictions = ca_tfgnn.process()
    print('predictions')
    print(predictions)


"""
  Loads training data.s
  Shuffles the training data.
  Takes a slice of the training data up to the side defined in data_set_size.
  This training set size can be reduced for pipeline experiments.

  @param data_set_size: number
  @return: dataframe, label array
"""


def collect_training_data(data_set_size):
    ag_news_ds = ds.load_dataset('ag_news', save_infos=True)
    ag_news_ds_train = ag_news_ds['train']
    ag_news_ds_train_df = pd.DataFrame(data=ag_news_ds_train)
    ag_news_ds_train_df_shuffled = ag_news_ds_train_df.sample(frac=1).reset_index(drop=True)
    ag_news_ds_train_df_shuffled = ag_news_ds_train_df_shuffled[:data_set_size]
    labels = ag_news_ds['test'].features['label'].names
    return ag_news_ds_train_df_shuffled, labels


if __name__ == "__main__":
    main()
