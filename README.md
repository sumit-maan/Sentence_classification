# sentence classification


# Data Set
Data set for sentence classification is taken from  Data folder repository on https://archive.ics.uci.edu/ml/datasets/Sentence+Classification

#Directories
1. labeled_articles: Contains data (90 text files) from 3 domains:
    PLOS, ARXIV, JDM
in these 90 text files there are sentences which are classified into 5 categories, labeled as below:
AIMX, OWNX, CONT, BASE, MISC

We divided this labeled_articles directory into 2 sets:
training set (contains 80% of labeled_articles)
test set (contains 20% of labeled_articles)

2. unlabeled_articles: contains 3 domains data out of which we used unlabled_JDM data set to test the sentence_classification model.

# Implementation

implemented Naive Bayes Classifier (NBC) using training data set to classify the sentences and used test data set to check the accuracy of model.

Find a text file "jdm_labeled.txt" which contains 5000 new sentences, which are labeled using implemented Naive Bayes Classifier


I had two techniques to solve the problem from which I choose Naive Bayes Classifier (NBC) over Decision Tree Classifier (DTC), the reason behind this is the overall time taken by NBC was too less compare to NTC , (though training time was less in case of DTC).
Also the accuracy is better in case of NBC .
