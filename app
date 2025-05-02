import numpy as np
import pandas as pd
import pyttsx3
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import threading
import os

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Global variables
model = None
scaler = None
df = None

def load_and_train_model():
    global model, scaler, df
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), "C:/Users/Janar/OneDrive/Desktop/New data set.csv .csv"))

        required_columns = ["GPS speed", "OBD speed", "Engine RPM", "Throttle position", 
                            "Engine load", "Coolant Temperature", "Fuel Consumption", 
                            "CO2 Emission", "Engine Temperature", "Brake Oil", "Maintenance Score"]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
            return

        X = df[["GPS speed", "OBD speed", "Engine RPM", "Throttle position", 
                "Engine load", "Coolant Temperature"]]
        y = df[["Fuel Consumption", "CO2 Emission", "Engine Temperature", 
                "Brake Oil", "Maintenance Score"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=40,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=5
            ),
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        result_label.config(text="Model trained successfully. Ready for evaluation.")
    except Exception as e:
        messagebox.showerror("Error", f"Auto-training failed: {e}")

def evaluate():
    if model is None:
        messagebox.showerror("Error", "Model not trained.")
        return

    try:
        entries = [
            ("GPS Speed", gps_speed_entry),
            ("OBD Speed", obd_speed_entry),
            ("Engine RPM", rpm_entry),
            ("Throttle Position", throttle_entry),
            ("Engine Load", load_entry),
            ("Coolant Temperature", coolant_temp_entry)
        ]
        
        user_input = []
        for name, entry in entries:
            value = entry.get().strip()
            if not value:
                messagebox.showerror("Error", f"Please enter {name}!")
                return
            try:
                user_input.append(float(value))
            except ValueError:
                messagebox.showerror("Error", f"Invalid number in {name}!")
                return

        user_input_scaled = scaler.transform([user_input])
        y_pred = model.predict(user_input_scaled)[0]

        fuel_consumption, co2_emission, engine_temp, brake_oil, maintenance_score = map(lambda x: round(x, 2), y_pred)

        if (maintenance_score > 80 and engine_temp < 90 and brake_oil < 30 and fuel_consumption < 10):
            engine_condition = "EXCELLENT ✅"
            solution = "Your engine is in perfect condition. Maintain current driving habits."
        elif (maintenance_score > 65 and engine_temp < 95 and brake_oil < 50 and fuel_consumption < 15):
            engine_condition = "GOOD 🌤"
            solution = "Engine is performing well. Schedule routine maintenance soon."
        elif (maintenance_score > 50 or engine_temp < 105 or brake_oil < 70):
            engine_condition = "MODERATE 🚨"
            solution = "Attention needed. Schedule maintenance and check engine parameters."
        else:
            engine_condition = "CRITICAL ❌"
            solution = "Immediate maintenance required! Engine at risk of damage."

        report = f"""Engine Health Report:
Condition: {engine_condition}
Fuel Consumption: {fuel_consumption} L/100km
CO2 Emission: {co2_emission} g/km
Engine Temperature: {engine_temp}°C
Brake Oil: {brake_oil}%
Maintenance Score: {maintenance_score}/100

Recommendation: {solution}"""

        result_label.config(text=report)
        engine.say(f"Engine condition is {engine_condition}. {solution}")
        engine.runAndWait()
    except Exception as e:
        messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

# GUI Setup
app = tk.Tk()
app.title("Vehicle Health Monitoring System")
app.geometry("800x800")
app.config(bg="#e3f2fd")

header = tk.Label(app, text="🚗 Vehicular Engine Health Monitor System", 
                  font=("Arial", 24, "bold"), bg="#42a5f5", fg="white", padx=20, pady=20)
header.pack(fill=tk.X)

# Input Fields Frame
input_frame = tk.Frame(app, bg="#e3f2fd")
input_frame.pack(pady=30)

fields = [
    ("GPS Speed (km/h)", "gps_speed_entry"),
    ("OBD Speed (km/h)", "obd_speed_entry"),
    ("Engine RPM", "rpm_entry"),
    ("Throttle Position (%)", "throttle_entry"),
    ("Engine Load (%)", "load_entry"),
    ("Coolant Temp (°C)", "coolant_temp_entry")
]

entry_widgets = []

for i, (label_text, var_name) in enumerate(fields):
    row = tk.Frame(input_frame, bg="#e3f2fd")
    label = tk.Label(row, text=label_text, font=("Arial", 14), bg="#e3f2fd", width=22, anchor="e")
    entry = tk.Entry(row, font=("Arial", 14), width=20, justify="center")
    
    label.pack(side=tk.LEFT, padx=5, pady=5)
    entry.pack(side=tk.LEFT, padx=5, pady=5)
    row.pack(anchor="center", pady=5)

    globals()[var_name] = entry
    entry_widgets.append(entry)

    # Bind Enter key to move to next field or evaluate
    def bind_return(e, idx=i):
        if idx < len(entry_widgets) - 1:
            entry_widgets[idx + 1].focus_set()
        else:
            evaluate()

    entry.bind("<Return>", bind_return)

# Evaluate Button
tk.Button(app, text="Evaluate Health", command=evaluate, 
          bg="#ffa726", fg="white", font=("Arial", 16, "bold"), width=20).pack(pady=30)

# Result Display
result_label = tk.Label(app, text="Loading and training model. Please wait...", 
                        font=("Arial", 14), bg="#e3f2fd", justify="left", wraplength=700)
result_label.pack(pady=20)

# Auto-load and train on start in a thread
threading.Thread(target=load_and_train_model, daemon=True).start()

# Start the GUI event loop
app.mainloop()
