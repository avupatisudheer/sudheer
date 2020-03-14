import pandas as pd
import numpy as np

# Loading the data set

cd = pd.read_csv("C:\\Users\\avupatisudheer\\Desktop\\Project Code\\Company_Description.csv")

cd.drop(["Unnamed: 0"], axis=1, inplace=True)
cd.columns = ["text","label"]

# Cleaning the data

import re

def cleaning_text1(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

cd.text = cd.text.apply(cleaning_text1)

# removing empty rows 

cd = cd.loc[cd.text != " ",:]

# removing leading white spaces and Punctuation

import string

def cleaning_text2(i):
    i = re.sub('^\s+','',i)
    i = "".join([c for c in i if c not in string.punctuation])
    i = re.sub('{company engaged business |company engaged providing |main income from }','',i)
    return i

cd.text = cd.text.apply(cleaning_text2)

 # Removing stopwords

with open("E:\\Assignments\\Text Mining\\stop.txt") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")
cd.text = [w for w in cd.text if not w in stopwords]

filename = 'Stop_words'
pickle.dump(stopwords, open(filename,'wb'))
Stop_model = pickle.load(open(filename,'rb'))

# splitting data into train and test data sets 
cd_accept = cd[cd['label']=='Accept']
cd_reject = cd[cd['label']=='Reject']
from sklearn.model_selection import train_test_split
train_accept,test_accept = train_test_split(cd_accept, test_size=0.2)
train_reject,test_reject = train_test_split(cd_reject, test_size=0.2)

cd_train = pd.concat([train_accept,train_reject], ignore_index=True)
cd_test = pd.concat([test_accept, test_reject], ignore_index=True)


# Preparing Document Term Matrix
    
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
cv = CountVectorizer()
data_cv = cv.fit(cd.text)
import pickle
pickle.dump(data_cv,open("NLP","wb"))
data_cv = pickle.load(open("NLP","rb"))

# For all messages
cd = data_cv.transform(cd.text)
cd.shape #(860,2716)
description = '''company is engaged in providing system integrations, VS channel andÂ is a global services delivery center. Main income is from Revenue from operations'''
X = data_cv.transform(list(description))
# For training Text
train_x = data_cv.transform(cd_train.text)
train_y = cd_train.label
train_x.shape #(687,2716)

# For test Text
test_x = data_cv.transform(cd_test.text)
test_y = cd_test.label
test_x.shape #(173,2716)

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.metrics import classification_report

# Multinomial Naive Bayes

classifier_mb = MB()
classifier_mb.fit(train_x,train_y)
train_pred_mb = classifier_mb.predict(train_x)
accuracy_train_mb = np.mean(train_pred_mb==train_y) # %94
confusion_matrix(train_y,train_pred_mb)
print(classification_report(train_y,train_pred_mb))

test_pred_mb = classifier_mb.predict(test_x)
accuracy_test_mb = np.mean(test_pred_mb==test_y) #  91%
confusion_matrix(test_y,test_pred_mb)

filename = 'NLP_model.pkl'
pickle.dump(classifier_mb, open(filename,'wb'))
model = pickle.load(open(filename,'rb'))
pred = model.predict(X)

d = {'des': [description]}
df = pd.DataFrame(d)
x = model.predict(data_cv.transform(df.des.apply(cleaning_text1)))
x

# Random forest on training dataset
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
rf.fit(train_x,train_y)
classifier = 'Random_Forest.pkl'
pickle.dump(rf, open(classifier,'wb'))
rf_model = pickle.load(open(classifier,'rb'))

d = {'des': [description]}
df = pd.DataFrame(d)
y = rf_model.predict(data_cv.transform(df.des.apply(cleaning_text1)))
y

# Training Accuracy
train_pred_rf = rf.predict(train_x)
accuracy_train_rf = np.mean(train_pred_rf==train_y) # 99%
confusion_matrix(train_y,train_pred_rf)

# Test Accuracy
test_pred_rf = rf.predict(test_x)
accuracy_test_rf = np.mean(test_pred_rf==test_y) # 93%
confusion_matrix(test_y,test_pred_rf)
