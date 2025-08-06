#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("D:/finalProject/phishingURLNew/pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)
@app.route("/")


@app.route("/login")
def login():
	return render_template('login.html')


@app.route("/home")
def home():
	return render_template('home.html')

@app.route("/contactus")
def contactus():
	return render_template('contactus.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/index2", methods=["GET", "POST"])
def index2():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30)

        y_pred =gbc.predict(x)[0]
        #1 is safe
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)