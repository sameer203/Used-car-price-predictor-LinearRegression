from flask import Flask, render_template,request, redirect
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('C:/Users/user/OneDrive/Documents/Data Science/Used_car_price_predictor/LinearRegressionModel.pkl','rb'))
car = pd.read_csv("C:/Users/user/OneDrive/Documents/Data Science/Used_car_price_predictor/Cleaned_car.csv")

@app.route('/')
def index():
    companies = sorted(car["Company"].unique())
    companies.insert(0, "select company")
    Models = sorted(car["Name"].unique())
    Year = sorted(car["Year"].unique(), reverse=True)
    Location = sorted(car["Location"].unique())
    Fuel = sorted(car["Fuel_Type"].unique())
    Owner = sorted(car["Owner_Type"].unique())
    Transmission = sorted(car["Transmission"].unique())
    return render_template("index.html", companies=companies, Models=Models, Year=Year, Location=Location, Fuel=Fuel, Owner=Owner, Transmission=Transmission)

@app.route("/predict", methods=["POST"])
def predict():

    company = request.form.get("company")
    car_model = request.form.get("Models")
    year = int(request.form.get("Year"))
    location = request.form.get("Location")
    fuel = request.form.get("Fuel")
    owner = request.form.get("Owner")
    transmission = request.form.get("Transmission")
    driven = request.form.get("Kilometers_Driven")

    print(company, car_model, year, location, fuel, owner, transmission, driven)

    prediction = model.predict(pd.DataFrame(columns=["Name", "Location","Year","Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type", "Company"],
                          data=np.array([car_model,location, year, driven, fuel, transmission, owner, company]).reshape(1,8)))

    print(prediction)
    return str(np.round(prediction[0],2))
    #return ""

if __name__ == "__main__":
    app.run(debug=True)
