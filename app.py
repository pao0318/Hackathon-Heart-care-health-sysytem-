import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/prediction', endpoint ='prediction')
def predicting():
    return render_template('predict5.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    p = model.predict_proba(final_features)
    prediction_chances=p[0][1]
    output=None
    if prediction == 0:
        output = "No"
    else:
        output = "Yes"


    return render_template('predict5.html', prediction_chances=prediction_chances )


if __name__ == "__main__":
    app.run(debug=True)