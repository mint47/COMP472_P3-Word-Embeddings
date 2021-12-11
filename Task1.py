import gensim.downloader as api
import pandas as pd
import csv

# Question 1

# load the word2vec-google-news-300 model  into the program
model = api.load("word2vec-google-news-300") # load data set as iterable

# load the data into synonym_df
synonym_df = pd.read_csv("synonyms.csv")

# create a question array
questions = synonym_df["question"]
# create answer array
answer = synonym_df["answer"]
# create first value array
firstValue = synonym_df["0"]
# create second value array
secondValue = synonym_df["1"]
# create third value array
thirdValue = synonym_df["2"]
# create fourth value array
fourthValue = synonym_df["3"]

# part 1
# print word2vec-google-news-300-details.csv file
with open('word2vec-google-news-300-details.csv', 'w') as f_object:
    # list of column names
    field_names = ['question', 'correct_answer', 'system_guess', 'labels']
    # create the csv writer
    writer = csv.writer(f_object, field_names)
    dw = csv.DictWriter(f_object, field_names)
    dw.writeheader()

    for x in range(0, len(synonym_df)):
        # convert tuple ques to string
        ques = questions[x]
        # convert tuple ans to string
        ans = answer[x]
        # convert tuple firstValue to string
        one = firstValue[x]
        # convert tuple secondValue to string
        two = secondValue[x]
        # convert tuple thirdValue to string
        three = thirdValue[x]
        # convert tuple fourthValue to string
        four = fourthValue[x]
        # in case the system's guess word is label as guess
        if (ques not in model.key_to_index) or (one not in model.key_to_index and two not in model.key_to_index and three not in model.key_to_index and four not in model.key_to_index):
            new_element = [ques, ans, '', 'guess']
            writer.writerow(new_element)
        # in case the system's guess word is label as correct
        elif (ques in model.key_to_index) and (one in model.key_to_index or two in model.key_to_index or three in model.key_to_index or four in model.key_to_index):
            guess = model.most_similar(ques)[0][0]
            if ans == guess:
                new_element = [ques, ans, guess, 'correct']
                writer.writerow(new_element)
            else:
                new_element = [ques, ans, guess, 'wrong']
                writer.writerow(new_element)


