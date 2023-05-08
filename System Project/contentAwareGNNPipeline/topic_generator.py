import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

from nltk.corpus import stopwords

"""
  Generates topics from each text sample using Linear Discriminant Analysis (LDA) 
  via a Gensim LDA Multicore Model
"""


class DocumentTopicGenerator():
    def __init__(self, data_df):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        self.data_df = data_df

    def update_df(self):
        self.data_df['text_topics'] = [self._predict_topics_from_gensim_lda_model(text) for text in
                                       self.data_df['text_preprocessed']]
        split = pd.DataFrame(self.data_df['text_topics'].to_list(),
                             columns=['text_topic_1', 'text_topic_2', 'text_topic_3'])
        return self._clean_up_df(pd.merge(self.data_df, split, how='left', left_index=True, right_index=True))

    """
      Removes tokens which are too long or short
    """

    def _gensim_preprocess(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def _remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in self.stop_words] for doc in texts]

    def _predict_topics_from_gensim_lda_model(self, text):
        data = [text]
        data_words = list(self._gensim_preprocess(data))
        data_words = self._remove_stopwords(data_words)
        id2word = corpora.Dictionary(data_words)
        texts = data_words
        corpus = [id2word.doc2bow(text) for text in texts]
        num_topics = 1
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)
        # Collect the topics from the first set provided in the list of topics.
        # A number of these are similar for this kind of corpus which is merely a
        # few sentences.
        topics = list(map(lambda tup: tup[1], lda_model.top_topics(corpus)[0][0]))
        # Just get the top 3 of the topics for now. It would be worth investigating
        # the effectiveness of adding additional topics.
        return topics[:3]

    def _clean_up_df(self, updated_pd):
        # Clean up the dataframe by imputing null values and removing columns that have been replaced
        topic_names = ['text_topic_1', 'text_topic_2', 'text_topic_3']
        for topic in topic_names:
            updated_pd[topic] = [n if n is not None else "" for n in updated_pd[topic]]
        used_columns = ['text_topics', 'text']
        for used_column in used_columns:
            updated_pd.drop([used_column], axis=1, inplace=True)
        return updated_pd
