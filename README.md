# COMP472_P3

Your Tasks
Write a program to use different word embeddings to answer the Synonym Test automatically and compare the
performance of different models. You must use:
1. Python 3.8 and the Gensim library. Gensim https://radimrehurek.com/gensim/ is a free open-source Python
library for representing documents as vectors. The library allows you to load pre-trained word embeddings,
train your own Word2Vec embeddings from your own corpus, computes the similarity of word embeddings, . . .
2. GitHub (make sure your project is private while developing).
Your work will be divided into 4 tasks.

Task 0: Contribution to a Collective Human Gold-Standard
This task is to be done individually. In order to better judge the performance of your code (tasks 1 to 3 below),
each of you will do the Synonym Test manually.
Before November 27, 2021 (11:59pm), go on the Moodle Quiz called Crowdsourced Gold-Standard for MP3
and answer each question to the best of your ability. Task 0 will count for 5% of your MP3 grade. You will
be graded on participation only, not on the correctness of your answers. Your grade will be proportional to the
number of questions that you answer.
After November 27, 2021, I will compile all the results and publish them on Moodle so that all teams can use
them as a human gold-standard to evaluate the results of your code (tasks 1 to 3 below).
2.2 Task 1: Evaluation of the word2vec-google-news-300 Pre-trained Model

Task 1:
In this first experiment, you will use the pre-trained Word2Vec model called word2vec-google-news-300 to
compute the closest synonym for each word in the dataset. First, use gensim.downloader.load to load the
word2vec-google-news-300 pretrained embedding model. Then use the similarity method from Gensim to
compute the cosine similarity between 2 embeddings (2 vectors) and find the closest synonym to the questionword.
The output of this task should be stored in 2 files:
1. In a file called <model name>-details.csv, for each question in the Synonym Test dataset, in a single line:
(a) the question-word, a comma,
(b) the correct answer-word, a comma
(c) your system’s guess-word, a comma
(d) one of 3 possible labels:
• the label guess, if either question-word or all four guess-words (or all 5 words) were not found in
the embedding model (so if the question-word was present in the model, and at least 1 guess-word
was present also, you should not use this label).
• the label correct, if the question-word and at least 1 guess-word were present in the model, and
the guess-word was correct.
• the label wrong if the question-word and at least 1 guess-word were present in the model, and the
guess-word was not correct.
For example, the file word2vec-google-news-300-details.csv could contain:
enormously,tremendously,uniquely,wrong
provisions,stipulations,stipulations,correct
...
2. In a file called analysis.csv, in a single line:
(a) the model name (clearly indicating the source of the corpus and the vector size), a comma
(b) the size of the vocabulary (the number of unique words in the corpus1)
(c) the number of correct labels (call this C), a comma
(d) the number of questions that your model answered without guessing (i.e. 80− guess) (call this V ), a comma
(e) the accuracy of the model (i.e. CV)
For example, the file analysis.csv could contain:
word2vec-google-news-300,3000000,44,78,0.5641025641025641


Task 2: Comparison with Other Pre-trained Models
Now that you have obtained results with the word2vec-google-news-300 pre-trained model, you will experiment with 4 other English word2vec pretrained models and compare the results. You can use any pre-trained
embeddings that you want, but you must have:
1. 2 new models from different corpora (eg. Twitter, English Wikipedia Dump . . . ) but same embedding size
(eg. 25, 100, 300)
2. 2 new models from the same corpus but different embedding sizes
Many pre-trained embeddings are available on line (including in Gensim or at http://vectors.nlpl.eu/repository).
For each model that you use, create a new <model name>-details.csv output file and append the results
to the file analysis.csv (see Section 2.2). For example, the file analysis.csv could now contain:
word2vec-google-news-300,3000000,44,78,0.5641025641025641 // from Task 1
C1-E1,...,...,...,...
C2-E2,...,...,...,...
C3-E3,...,...,...,...
C4-E4,...,...,...,...
where C1 to C4 refer to the corpora and E1 to E4 refer to their embedding sizes, and C1 ̸= C2 and E1 = E2
and C3 ̸= C4 and E3 = E4.
Compare the performance of these models (graphs would be very useful here) and compare them to a random
baseline and a human gold-standard. Analyse your data points and speculate on why some model perform better
than others.
