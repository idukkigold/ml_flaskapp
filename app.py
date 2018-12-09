from flask import Flask, render_template,request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/get-user-data', methods=['POST'])
def predict_stuff():
    if request.method == 'POST':
        model = joblib.load('trained_anomaly detection.pkl')

        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        height = int(request.form.get('height'))
        weight = int(request.form.get('weight'))
        QRS_duration = int(request.form.get('QRS_duration'))
        PR = int(request.form.get('PR'))
        QT = int(request.form.get('QT'))
        T = int(request.form.get('T'))
        P = request.form.get('P')
        aQRST = request.form.get('aQRST')
        J = request.form.get('J')

        heart_rate = request.form.get('heart_rate')

        wQ = request.form.get('wQ')

        wR = request.form.get('wR')

        wS = request.form.get('wS')
        wR_ = request.form.get('wR_')

        webdata = [age,sex,height,weight,QRS_duration,PR,QT,T,P,aQRST,J, heart_rate,wQ,wR, wS ,wR_,1,1,1,1,1,1]


        webdata = [
            webdata
        ]

        # return render_template("index.html", pred=house_to_value) 

        # Run the model and make a prediction for each house in the homes_to_value array
        predicted_value = model.predict(webdata)

        # Since we are only predicting the price of one house, just look at the first prediction returned
        predicted_value = predicted_value[0]
        print(predicted_value)
        return render_template("index.html", pred=predicted_value)

if __name__ == "__main__":
    app.run()
