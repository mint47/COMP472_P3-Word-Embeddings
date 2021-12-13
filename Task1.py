import gensim.downloader as api
import pandas as pd
import csv
import os

from pandas.core.frame import DataFrame

# Question 1
# load the word2vec-google-news-300 model  into the program
print("> loading model...")
model_name = "word2vec-google-news-300"
def load_model(model_name):
    return api.load(model_name) # load data set as iterable
google300 = load_model(model_name)
print(" >> model loaded")

print("> loading questions...")
def load_questions():
    # load the data into synonym_df
    synonym_df = pd.read_csv("synonyms.csv")
    # create a question array
    questions = synonym_df["question"]
    # create answer array
    answers = synonym_df["answer"]
    # create options array
    options = pd.DataFrame(synonym_df, columns=['0', '1', '2', '3'])
    options = options.transpose()
    return questions, answers, options
questions, answers, options = load_questions()
print(" >> questions loaded")

# part 1
# print word2vec-google-news-300-details.csv file
print("> going through questionaire...")
def findmostsimilar(questions, answers, options, model, output):
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
                print('Guess:'+guess)

                label = 'guess'
                
            # in case the system's guess word is label as correct
            else:
                # compute cosine similarity of options
                sim = [model.similarity(ques, o) for o in opt if o in model.key_to_index]
                # choose most similar option
                guess = opt[sim.index(max(sim))]
                print('Guess:'+guess)

                if ans == guess:
                    label = 'correct'
                else:
                    label = 'wrong'
            writer.writerow([ques, ans, guess, label])
findmostsimilar(questions, answers, options, google300, 'word2vec-google-news-300-details.csv')
print(" >> all questions answered")

# part 2
print("> getting analytics")
def getAnalytics(model, model_name):
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
    accuracy = float(correct_labels) / without_guessing

    return model_name, emb_size, vocabulary_size, correct_labels, without_guessing, accuracy
model_name, emb_size, vocabulary_size, correct_labels, without_guessing, accuracy = getAnalytics(google300, model_name)

print(" >> saving analytics to file")
def saveAnalysis(model_name, vocabulary_size, correct_labels, without_guessing, accuracy):
    if (not os.path.isfile('analysis.csv')):
        # list of column name
        field_names = ['model name', 'vocabulary size', 'correct labels', 'model answered without guessing', 'accuracy']
        file = DataFrame(columns=field_names)
        file.to_csv('analysis.csv', index=False)
    with open('analysis.csv', 'a', newline='\n') as file:
        analysis_value = [model_name, vocabulary_size, correct_labels, without_guessing, accuracy]
        writer = csv.writer(file)
        writer.writerow(analysis_value)
saveAnalysis(model_name, vocabulary_size, correct_labels, without_guessing, accuracy)
print(" >> finished saving analytics")
print("> end of program")