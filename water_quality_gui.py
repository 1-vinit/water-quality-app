# ============================================================== 
# 🧠 AI-Based Water Quality Prediction for Sewage Treatment Plant
# ============================================================== 
# Author: Vinit [B.Tech (CSE), MIET]
# Description: ML + GUI project for college submission
# ==============================================================

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ============================================================ 
# STEP 0: Generate Synthetic Dataset
# ============================================================ 
np.random.seed(42)
data = []

for _ in range(300):
    pH = np.round(np.random.uniform(5.0, 9.0), 2)
    bod = np.round(np.random.uniform(1, 10), 2)
    cod = np.round(np.random.uniform(10, 60), 2)
    tds = np.round(np.random.uniform(200, 1200), 2)

    # Labeling rules
    if 6.5 <= pH <= 8.5 and bod <= 3 and cod <= 20 and tds <= 500:
        quality = "Safe"
    elif (bod <= 6 and cod <= 40 and tds <= 1000) and not (6.5 <= pH <= 8.5 and bod <= 3 and cod <= 20 and tds <= 500):
        quality = "Needs Treatment"
    else:
        quality = "Unfit"

    data.append([pH, bod, cod, tds, quality])

df = pd.DataFrame(data, columns=["pH", "BOD", "COD", "TDS", "Quality"])

# ============================================================ 
# STEP 1: Train Model
# ============================================================ 
X = df[["pH", "BOD", "COD", "TDS"]]
y = df["Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Save the model
joblib.dump(model, "water_quality_model.pkl")

print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# ============================================================ 
# STEP 2: Tkinter GUI Functions
# ============================================================ 

def predict_quality():
    try:
        pH = float(entry_ph.get())
        bod = float(entry_bod.get())
        cod = float(entry_cod.get())
        tds = float(entry_tds.get())

        features = pd.DataFrame([[pH, bod, cod, tds]], columns=["pH", "BOD", "COD", "TDS"])
        prediction = model.predict(features)[0]

        if prediction == "Safe":
            result_label.config(text="✅ Water is Safe for reuse", fg="green")
        elif prediction == "Needs Treatment":
            result_label.config(text="⚠️ Water Needs Treatment", fg="orange")
        else:
            result_label.config(text="❌ Water is Unfit", fg="red")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

def show_accuracy():
    messagebox.showinfo("Model Accuracy", f"Model Accuracy: {acc*100:.2f}%")

def show_chart():
    counts = df["Quality"].value_counts()
    plt.figure(figsize=(5,4))
    bars = plt.bar(counts.index, counts.values, color=["green","orange","red"])
    plt.title("Class Distribution")
    plt.xlabel("Water Quality")
    plt.ylabel("Count")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+0.1, yval+1, yval)
    plt.show()

# ============================================================ 
# STEP 3: Build Tkinter GUI
# ============================================================ 

root = tk.Tk()
root.title("AI-Based Water Quality Prediction")
root.geometry("450x520")
root.configure(bg="#E6F0FF")

tk.Label(root, text="AI-Based Water Quality Prediction", font=("Arial", 14, "bold"), bg="#E6F0FF", fg="#003366").pack(pady=15)

# Input boxes
tk.Label(root, text="pH:", bg="#E6F0FF", font=("Arial", 11)).pack()
entry_ph = tk.Entry(root, font=("Arial", 11))
entry_ph.pack(pady=2)

tk.Label(root, text="BOD (mg/L):", bg="#E6F0FF", font=("Arial", 11)).pack()
entry_bod = tk.Entry(root, font=("Arial", 11))
entry_bod.pack(pady=2)

tk.Label(root, text="COD (mg/L):", bg="#E6F0FF", font=("Arial", 11)).pack()
entry_cod = tk.Entry(root, font=("Arial", 11))
entry_cod.pack(pady=2)

tk.Label(root, text="TDS (mg/L):", bg="#E6F0FF", font=("Arial", 11)).pack()
entry_tds = tk.Entry(root, font=("Arial", 11))
entry_tds.pack(pady=2)

# Buttons
tk.Button(root, text="🔍 Predict Quality", command=predict_quality, bg="#007BFF", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
tk.Button(root, text="📊 Show Accuracy", command=show_accuracy, bg="#28A745", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(root, text="📈 Show Class Distribution", command=show_chart, bg="#FFC107", fg="black", font=("Arial", 12, "bold")).pack(pady=5)

# Result Label
result_label = tk.Label(root, text="", bg="#E6F0FF", font=("Arial", 13, "bold"))
result_label.pack(pady=10)

# Footer
tk.Label(root, text=f"Model Accuracy: {acc*100:.2f}%", bg="#E6F0FF", fg="blue", font=("Arial", 10, "italic")).pack(pady=5)
tk.Label(root, text="Developed by Vinit\nDept. of B.Tech (CSE), MIET", bg="#E6F0FF", fg="blue", font=("Arial", 10)).pack(side="bottom", pady=10)

root.mainloop()
