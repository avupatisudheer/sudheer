from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import re

# Load the model from disk
classifier = 'Random_Forest.pkl'
rf_model = pickle.load(open(classifier,'rb'))
data_cv = pickle.load(open("NLP","rb"))
app = Flask(__name__,template_folder='C:\\Users\\avupatisudheer\\Desktop\\Company Description\\templates')

def cleaning_text1(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

def cleaning_text2(i):
    i = re.sub('^\s+','',i)
    i = "".join([c for c in i if c not in string.punctuation])
    i = re.sub('{company engaged business |company engaged providing |main income from }','',i)
    return i

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        d = {'des': [message,]}
        df = pd.DataFrame(d)
        y = rf_model.predict(data_cv.transform(df.des.apply(cleaning_text1)))
        print(y[0])
        value = y[0]
    return render_template('result.html',prediction = value)

if __name__ == '__main__':
	app.run(debug=True)
 