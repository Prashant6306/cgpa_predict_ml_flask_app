from flask import Flask, render_template, request
import joblib
import numpy as np
app=Flask(__name__)
import sklearn
rg=open("cgpa_regression_model.pkl","rb")
ml_model=joblib.load(rg)


@app.route("/")
def test():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        print(request.form.get('cgpa'))
        try:
            cgpa=float(request.form['cgpa'])
            pred_args = [cgpa]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            #mul_reg = open("cgpa_regression_model.pkl", "rb")
            #ml_model = joblib.load(mul_reg)
            #model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = ml_model.predict(pred_args_arr)
            round(float(model_prediction), 2)
        except:
            return "check your value"
    return render_template("predict.html",prediction=model_prediction)



if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')