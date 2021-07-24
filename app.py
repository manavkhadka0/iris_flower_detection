from flask import Flask, render_template,request
import pickle
import numpy as np
from iris import accuracy,set_matrix,ver_matrix,vir_matrix

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def pred():
    data1 = request.form['sl']
    data2 = request.form['sw']
    data3 = request.form['pl']
    data4 = request.form['pw']
    arr = np.array([[data1,data2,data3,data4]])
    output = model.predict(arr)
    return render_template('output.html',data=output)


@app.route("/info")
def info():
    return render_template('info.html',acc=accuracy,set=set_matrix,ver=ver_matrix,vir=vir_matrix)

if __name__ == '__main__':
    app.run(debug=True)