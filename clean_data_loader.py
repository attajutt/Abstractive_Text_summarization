from imports import *

rf_pkl = open('./cleaned_data/clean_summaries.pkl', 'rb')
clean_summaries=cPickle.load(rf_pkl)
rf_pkl.close()

rf_pkl = open('./cleaned_data/clean_texts.pkl', 'rb')
clean_texts=cPickle.load(rf_pkl)
rf_pkl.close()


rf_pkl = open('./cleaned_data/sorted_summaries_ints.pkl', 'rb')
sorted_summaries=cPickle.load(rf_pkl)
rf_pkl.close()

rf_pkl = open('./cleaned_data/sorted_texts_ints.pkl', 'rb')
sorted_texts=cPickle.load(rf_pkl)
rf_pkl.close()


rf_pkl = open('./cleaned_data/vocab_to_int.pkl', 'rb')
vocab_to_int=cPickle.load(rf_pkl)
rf_pkl.close()

rf_pkl = open('./cleaned_data/int_to_vocab.pkl', 'rb')
int_to_vocab=cPickle.load(rf_pkl)
rf_pkl.close()

rf_pkl = open('./cleaned_data/word_embeddings.pkl', 'rb')
word_embedding_matrix=cPickle.load(rf_pkl)
rf_pkl.close()

