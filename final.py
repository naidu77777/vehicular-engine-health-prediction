from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import pyttsx3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Setup Flask app
app = Flask(__name__, template_folder='../templates')
CORS(app)

# ✅ Absolute path to dataset
DATA_PATH = "C:/Users/Janar/OneDrive/Desktop/final project/data/engine_health_enhanced.csv"
MODEL_PATH = "./models/"
FEATURE_COLUMNS = []

TARGETS = ["Fuel_Consumption", "CO2_Emission", "Engine_Temp", "Brake_Wear", "Maintenance_Score"]

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def train_and_save_models():
    global FEATURE_COLUMNS

    print("⏳ Training models... please wait.")
    data = pd.read_csv(DATA_PATH)
    data = data.sample(n=5000, random_state=42)

    label_encoder = LabelEncoder()
    data["Condition"] = label_encoder.fit_transform(data["Condition"])
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"), compress=3)

    X = data.drop(columns=["Condition"] + TARGETS)
    y = data["Condition"]
    FEATURE_COLUMNS = list(X.columns)

    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_PATH, "feature_columns.pkl"), compress=3)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"), compress=3)

    base_models = {
        "DecisionTree": DecisionTreeRegressor(max_depth=10),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50),
        "RandomForest": RandomForestRegressor(n_estimators=30, n_jobs=-1, max_depth=10),
        "SVR": SVR(kernel='linear', C=0.5),
        "AdaBoost": AdaBoostRegressor(n_estimators=50)
    }

    train_meta = []
    for name, model in base_models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, os.path.join(MODEL_PATH, f"{name}_model.pkl"), compress=3)
        train_meta.append(model.predict(X_train_scaled))

    X_train_meta = np.vstack(train_meta).T
    meta_model = LinearRegression()
    meta_model.fit(X_train_meta, y_train)
    joblib.dump(meta_model, os.path.join(MODEL_PATH, "Stacked_model.pkl"), compress=3)

    for target in TARGETS:
        y_target = data.loc[X_train.index, target]
        model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_target)
        joblib.dump(model, os.path.join(MODEL_PATH, f"{target}_model.pkl"), compress=3)

    print("✅ All models trained and saved.")

@app.route('/')
def serve_html():
    return render_template('thtml.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global FEATURE_COLUMNS
        if not FEATURE_COLUMNS:
            FEATURE_COLUMNS = joblib.load(os.path.join(MODEL_PATH, "feature_columns.pkl"))

        input_data = request.get_json()
        input_values = [float(input_data[col]) for col in FEATURE_COLUMNS]

        scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
        input_scaled = scaler.transform([input_values])

        base_model_names = ["DecisionTree", "GradientBoosting", "RandomForest", "SVR", "AdaBoost"]
        base_preds = []
        for name in base_model_names:
            model = joblib.load(os.path.join(MODEL_PATH, f"{name}_model.pkl"))
            base_preds.append(model.predict(input_scaled)[0])

        meta_input = np.array(base_preds).reshape(1, -1)
        stacked_model = joblib.load(os.path.join(MODEL_PATH, "Stacked_model.pkl"))
        final_pred = stacked_model.predict(meta_input)[0]
        condition = label_encoder.inverse_transform([int(round(final_pred))])[0]

        predictions = {}
        for target in TARGETS:
            model = joblib.load(os.path.join(MODEL_PATH, f"{target}_model.pkl"))
            predictions[target] = round(model.predict(input_scaled)[0], 2)

        report = (
            f"Fuel Consumption is {predictions['Fuel_Consumption']} liters per 100 kilometers. "
            f"CO2 Emission is {predictions['CO2_Emission']} grams per kilometer. "
            f"Engine Temperature is {predictions['Engine_Temp']} degrees Celsius. "
            f"Brake Wear is {predictions['Brake_Wear']} percent. "
            f"Maintenance Score is {predictions['Maintenance_Score']}. "
        )

        if condition == "CRITICAL":
            advice = "🔴 Engine Condition: CRITICAL 🚨\n💡 Visit a mechanic immediately."
            report += "Engine Condition is Critical. Visit a mechanic immediately."
        elif condition == "MODERATE":
            advice = "🟠 Engine Condition: MODERATE ⚠️\n💡 Schedule a maintenance check."
            report += "Engine Condition is Moderate. Schedule a maintenance check."
        else:
            advice = "🟢 Engine Condition: GOOD ✅\n💡 Everything looks fine."
            report += "Engine Condition is Good. Everything looks fine."

        speak_text(report)

        return jsonify({
            "fuel": predictions["Fuel_Consumption"],
            "co2": predictions["CO2_Emission"],
            "temp": predictions["Engine_Temp"],
            "brake": predictions["Brake_Wear"],
            "score": predictions["Maintenance_Score"],
            "advice": advice
        })

    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    if not os.path.exists(os.path.join(MODEL_PATH, "Stacked_model.pkl")):
        train_and_save_models()
    app.run(debug=True)