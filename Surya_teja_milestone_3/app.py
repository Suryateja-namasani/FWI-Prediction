from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open("ridge.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    FFMC = float(request.form["FFMC"])
    DMC = float(request.form["DMC"])
    DC = float(request.form["DC"])
    ISI = float(request.form["ISI"])
    Temperature = float(request.form["Temperature"])
    Ws = float(request.form["Ws"])
    BUI = float(request.form["BUI"])

    # Prepare input and scale
    data = np.array([[FFMC, DMC, DC, ISI, Temperature, Ws, BUI]])
    scaled = scaler.transform(data)

    # Prediction
    prediction = round(model.predict(scaled)[0], 3)

    # Fire danger classification
    if prediction < 5:
        level = "🟢 LOW Fire Danger"
        image_file = "low.png"
    elif prediction < 12:
        level = "🟡 MODERATE Fire Danger"
        image_file = "moderate.png"
    elif prediction < 21:
        level = "🟠 HIGH Fire Danger"
        image_file = "high.png"
    elif prediction < 38:
        level = "🔴 VERY HIGH Fire Danger"
        image_file = "veryhigh.png"
    else:
        level = "🔥 EXTREME Fire Danger"
        image_file = "extreme.png"

    return render_template("result.html",
                           result=prediction,
                           level=level,
                           image_file=image_file)

if __name__ == "__main__":
    app.run(debug=True)
