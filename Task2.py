import gensim.downloader as api
import pandas as pd
import csv
import Task1

# load questions
questions, answers, options = Task1.load_questions()

# using: print(list(api.info()['models'].keys()))
# available models found are:
# ['fasttext-wiki-news-subwords-300', 
#  'conceptnet-numberbatch-17-06-300', 
#  'word2vec-ruscorpora-300', 
#  'word2vec-google-news-300', 
#  'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 
#  'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 
#  '__testing_word2vec-matrix-synopsis']

# experiment with 4 other English word2vec pretrained models and compare the results.
# you must have:
#   1. 2 new models from different corpora (eg. Twitter, English Wikipedia Dump . . . ) but same embedding size (eg. 25, 100, 300)
#   2. 2 new models from the same corpus but different embedding sizes

# 2.1
wiki50 = Task1.load_model('glove-wiki-gigaword-50')
twitter50 = Task1.load_model('glove-twitter-50')