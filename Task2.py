import gensim.downloader as api
import pandas as pd
from pandas.core.frame import DataFrame
import csv
import os

# load questions
question_file = "synonyms.csv"
def load_questions(question_file):
    # load the data into synonym_df
    synonym_df = pd.read_csv(question_file)
    # create a question array
    questions = synonym_df["question"]
    # create answer array
    answers = synonym_df["answer"]
    # create options array
    options = pd.DataFrame(synonym_df, columns=['0', '1', '2', '3'])
    options = options.transpose()
    return questions, answers, options
questions, answers, options = load_questions(question_file)

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
def findmostsimilar2(questions, answers, options, model_name):
    print("> loading model...")
    model = api.load(model_name)
    print("> model loaded")
    print("> creating output file...")
    output = (model_name+'-details.csv')
    print("> finding synonyms...")
    with open(file=output, mode='w', newline='\n') as file:
        # list of column names
        field_names = ['question', 'correct_answer', 'system_guess', 'labels']
        # create the csv writer
        writer = csv.writer(file, field_names)
        writer.writerow(field_names)

        for x in range(0, len(questions)):
            # convert tuple ques to string
            ques = questions[x]
            print('Question: '+ques)
            # convert tuple ans to string
            ans = answers[x]
            # get options for question
            opt = options[x]

            # in case the system's guess word is label as guess
            if (ques not in model.key_to_index) or (opt[0] not in model.key_to_index and opt[1] not in model.key_to_index and opt[2] not in model.key_to_index and opt[3] not in model.key_to_index):
                guess = opt[0]
                label = 'guess'
                
            # in case the system's guess word is label as correct
            else:
                # compute cosine similarity of options
                sim = [model.similarity(ques, o) for o in opt if o in model.key_to_index]
                # choose most similar option
                guess = opt[sim.index(max(sim))]
                if ans == guess:
                    label = 'correct'
                else:
                    label = 'wrong'
            print('Guess:'+guess)
            writer.writerow([ques, ans, guess, label])
    return model
def saveAnalytics(model, model_name):
    # model name
    model_name = model_name
    # embeding size
    emb_size = model.vector_size
    # vocabulary size
    vocabulary_size = len(model)
    
    # loading word2vec-google-news-300-details.csv into pandas
    output_file = (model_name+'-details.csv')
    df = pd.read_csv(output_file)
    # correct labels
    correct_labels = 0
    # number of questions answered without guessing
    without_guessing = 80

    for x in df['labels']:
        if x == 'guess':
            without_guessing = without_guessing - 1
        elif x == 'correct':
            correct_labels = correct_labels + 1
    # accuracy
    if without_guessing != 0:
        accuracy = float(correct_labels) / without_guessing
    else:
        accuracy = 0

    print("> saving analytics to file...")
    if (not os.path.isfile('analysis.csv')):
        # list of column name
        field_names = ['model name', 'vocabulary size', 'correct labels', 'model answered without guessing', 'accuracy']
        file = DataFrame(columns=field_names)
        file.to_csv('analysis.csv', index=False)
    with open('analysis.csv', 'a', newline='\n') as file:
        analysis_value = [model_name, vocabulary_size, correct_labels, without_guessing, accuracy]
        writer = csv.writer(file)
        writer.writerow(analysis_value)
    print("> done!")

model_names = ['fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300', 'glove-twitter-100', 'glove-twitter-200']

model1 = findmostsimilar2(questions, answers, options, model_names[0])
saveAnalytics(model1, model_names[0])

model2 = findmostsimilar2(questions, answers, options, model_names[1])
saveAnalytics(model2, model_names[1])

model3 = findmostsimilar2(questions, answers, options, model_names[2])
saveAnalytics(model3, model_names[2])

model4 = findmostsimilar2(questions, answers, options, model_names[3])
saveAnalytics(model4, model_names[3])
