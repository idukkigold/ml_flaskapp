from flask import Flask, render_template, jsonify, redirect, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html") 


@app.route('/get-user-data', methods=['POST'])
def predict_stuff():
    if request.method == 'POST':
        model = joblib.load('trained_house_classifier_model.pkl')

        print('-----line 27--------')
        print(request.form.get('year_built'))

        year_built = int(request.form.get('year_built'))

        print('line 31')

        stories = int(request.form.get('stories'))
        num_bedrooms = int(request.form.get('num_bedrooms'))
        full_bathrooms = int(request.form.get('full_bathrooms'))
        half_bathrooms = int(request.form.get('half_bathrooms'))
        livable_sqft = int(request.form.get('livable_sqft'))
        total_sqft = int(request.form.get('total_sqft'))
        garage_sqft = int(request.form.get('garage_sqft'))

        carport_sqft = int(request.form.get('carport_sqft'))
        has_fireplace = request.form.get('has_fireplace')

        has_pool = request.form.get('has_pool')

        has_central_heating = request.form.get('has_central_heating')

        has_central_cooling = request.form.get('has_central_cooling')


        has_fireplace = request.form.get('has_fireplace')

        garage_type = request.form.get('garage_type')
 
        city = request.form.get('city')

        house_to_value = [
            # House features
            year_built,   # year_built
            stories,      # stories
            num_bedrooms,      # num_bedrooms
            full_bathrooms,      # full_bathrooms
            half_bathrooms,      # half_bathrooms 
            livable_sqft,   # livable_sqft
            total_sqft,   # total_sqft
            garage_sqft,      # garage_sqft
            carport_sqft,      # carport_sqft

        ]

        # scikit-learn assumes you want to predict the values for lots of houses at once, so it expects an array.
        # We just want to look at a single house, so it will be the only item in our array.
        homes_to_value = [
            house_to_value
        ]

        # return render_template("index.html", pred=house_to_value) 

        # Run the model and make a prediction for each house in the homes_to_value array
        predicted_home_values = model.predict(homes_to_value)

        # Since we are only predicting the price of one house, just look at the first prediction returned
        predicted_value = predicted_home_values[0]

        return render_template("index.html", pred=predicted_value) 

if __name__ == "__main__":
    app.run()
