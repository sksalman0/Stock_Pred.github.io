from flask import Flask,render_template,request,jsonify
import pandas as pd
from single import model,m_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')   
@app.route('/predict',methods=['POST'])
def predict_price():
    stock_name = str(request.form.get('sn')).capitalize()
    Open,High,Low,pred = model(stock_name)
    return render_template('index.html',prediction = f'Tomorrows Closing Price:{pred}')
@app.route('/m_predict',methods=['POST'])
def mpredict():
    stock_name = str(request.form.get('sn1')).capitalize()
    Open = float(request.form.get('op'))
    High = float(request.form.get('hi'))
    Low = float(request.form.get('lo'))
    Volum = float(request.form.get('vo'))
    pred =  m_model(stock_name, Open, High, Low,Volum)
    return render_template('index.html',prediction=f'Tomorrows Closing price for given data is {pred}')
    



if __name__ == '__main__':
    app.run(debug=True)