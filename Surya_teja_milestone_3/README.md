Project Title
Forest Fire Weather Index Prediction using a Ridge Regression model.

Overview
This project predicts the Fire Weather Index using seven input features.
The app uses a trained Ridge model and a scaler.
The frontend uses two HTML templates.
The backend uses Flask.

Features
• Input fields for FFMC, DMC, DC, ISI, Temperature, Ws, BUI
• Scaled model prediction
• Fire danger classification
• Result page with danger level and an image
• Simple gradient UI

Tech Stack
• Python
• Flask
• NumPy
• Scikit-learn
• HTML and CSS
• Pickle for model loading

Project Structure
• app.py holds all backend routes
• templates/index.html is the input page
• templates/result.html shows the output
• static contains images and background assets
• ridge.pkl stores the trained model
• scaler.pkl stores the fitted scaler

How to Run

Install Python 3

Install the required packages
pip install flask numpy scikit-learn

Place ridge.pkl and scaler.pkl in the project folder

Run the app
python app.py

Open the browser at
http://127.0.0.1:5000

Model Inputs
• FFMC
• DMC
• DC
• ISI
• Temperature
• Wind speed
• BUI

Output
• FWI numeric value
• Fire danger category
Low
Moderate
High
Very High
Extreme
 