import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox
import os
import sys

import os
import joblib

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型和编码器
model_path = os.path.join(current_dir, 'model_classifier.pkl')
encoder_path = os.path.join(current_dir, 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)


# Define the prediction function
def predict_model():
    try:
        # Get user input
        sale_price = float(entry_price.get())
        electric_range = float(entry_range.get())

        # Check if inputs are within the valid range
        if not (10000 <= sale_price <= 100000):
            messagebox.showerror("Input Error", "Sale Price must be between $10,000 and $100,000.")
            return
        if not (1 <= electric_range <= 500):
            messagebox.showerror("Input Error", "Electric Range must be between 1 km and 500 km.")
            return

        # Prepare input data
        new_data = pd.DataFrame({'Sale Price': [sale_price], 'Electric Range': [electric_range]})

        # Predict the model
        predicted_model_encoded = model.predict(new_data)
        predicted_model = label_encoder.inverse_transform(predicted_model_encoded)

        # Show prediction result
        messagebox.showinfo("Prediction Result", f"Predicted Model: {predicted_model[0]}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create the main window
app = tk.Tk()
app.title("Electric Vehicle Model Predictor")

# Create labels and input fields
tk.Label(app, text="Sale Price ($, 10,000-100,000):").grid(row=0, column=0, padx=10, pady=5)
entry_price = tk.Entry(app)
entry_price.grid(row=0, column=1, padx=10, pady=5)

tk.Label(app, text="Electric Range (km, 1-500):").grid(row=1, column=0, padx=10, pady=5)
entry_range = tk.Entry(app)
entry_range.grid(row=1, column=1, padx=10, pady=5)

# Create a predict button
btn_predict = tk.Button(app, text="Predict Model", command=predict_model)
btn_predict.grid(row=2, column=0, columnspan=2, pady=10)

# Create an exit button
btn_exit = tk.Button(app, text="Exit", command=app.quit)
btn_exit.grid(row=3, column=0, columnspan=2, pady=5)

# Run the main loop
app.mainloop()
