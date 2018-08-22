import os
import glob, time
import nltk.classify
# nltk.download("punkt")
from nltk import word_tokenize
import json
import re
import textblob
from textblob.classifiers import NaiveBayesClassifier as nbc

# extract all the text files from a directry
def txt_file(text_files_dir):
    n = os.listdir(text_files_dir)
    for i in range(len(n)):
        text_files = []
        for file1 in glob.iglob(text_files_dir + "/*.txt".format(i+1)):
            text_files.append(file1)
    return text_files

#reading text files, returns labels and sentences from lines
def txt_to_sentence(list_of_text_files):
    label = []
    sentence = []
    for txt_file in list_of_text_files:
        f = open(txt_file, "r")
        lines = f.readlines()
        for sent in lines:
            if sent.startswith("#"):
                continue
            elif "\t" in sent:
                label.append(sent.split("\t")[0])
                sentence.append(sent.split("\t")[1].lower())
            else:
                label.append(sent.split(" ")[0])
                sentence.append(sent[len(sent.split(" ")[0])+1:].lower())
    for i in range(len(sentence)):
        if sentence[i].endswith("\n"):
            sentence[i] = sentence[i][:-1]
        sentence[i] = " ".join([word for word in word_tokenize(sentence[i])])
        sentence[i] = re.sub(' +', ' ', sentence[i])
    return label, sentence

#Prepare training and testing dataset (80:20 % in our case)
path = "/home/ubuntu/SentenceCorpus/labeled_articles"
text_files = txt_file(path)
training_data = text_files[0:round(len(text_files)*.8)]
test_data = text_files[round(len(text_files)*.8):]
#write training data into a json file
label, sentence = txt_to_sentence(training_data)
data_t = []
for i in range(len(sentence)):
    data = {'label': label[i], "text": sentence[i] }
    data_t.append(data)
f_name = "/home/ubuntu/SentenceCorpus" + "/" + "training" + ".json"
with open(f_name, 'w') as f:
    json.dump(data_t, f,ensure_ascii = True)

#Naive Bayes Classifier for sentence classification
with open(f_name, 'r') as f:
    nb = nbc(f, format="json")

#validation using test dataset
test_l, test_s = txt_to_sentence(test_data)
counter = 0
for i in range(len(test_s)):
    predicted_label = nb.classify(test_s[i]).upper()
    original_label= (test_l[i]).upper()
    if (predicted_label==original_label):
        counter += 1

print("Accuracy: " + str(round(counter*100/len(test_s), 2)) + "%")
print("cheers")
