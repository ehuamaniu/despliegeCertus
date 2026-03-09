from flask import Flask, request, jsonify
import pandas as pd
import pickle
import  os

app = Flask(__name__)

# Cargar modelo al iniciar
with open("modelo_diabetes.pkl", "rb") as archivo:
    modelo = pickle.load(archivo)

columnas = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

@app.route("/")
def home():
    return "API de predicción de diabetes activa"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        datos = request.get_json()

        paciente = pd.DataFrame([[
            datos["Pregnancies"],
            datos["Glucose"],
            datos["BloodPressure"],
            datos["SkinThickness"],
            datos["Insulin"],
            datos["BMI"],
            datos["DiabetesPedigreeFunction"],
            datos["Age"]
        ]], columns=columnas)

        prediccion = int(modelo.predict(paciente)[0])

        resultado = "El paciente tiene diabetes" if prediccion == 1 else "El paciente no tiene diabetes"

        return jsonify({
            "prediccion": prediccion,
            "resultado": resultado
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto)