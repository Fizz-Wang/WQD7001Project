import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the model and encoder
model = joblib.load('model_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the prediction function
def predict_model():
    try:
        # Get user input
        sale_price = float(entry_price.get())
        electric_range = float(entry_range.get())

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
tk.Label(app, text="Sale Price ($):").grid(row=0, column=0, padx=10, pady=5)
entry_price = tk.Entry(app)
entry_price.grid(row=0, column=1, padx=10, pady=5)

tk.Label(app, text="Electric Range (km):").grid(row=1, column=0, padx=10, pady=5)
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
