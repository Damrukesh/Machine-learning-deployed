import pickle
import numpy as np
from flask import Flask,request,jsonify,render_template
app=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=request.form
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction=model.predict(new_data)
    return jsonify(prediction[0])
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=[float(x) for x in request.form.values()]
    new_data=scaler.transform(np.array(data).reshape(1,-1))
    prediction=model.predict(new_data)
    return render_template('index.html',prediction_text="The predicted price is {}".format(prediction[0]))
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
