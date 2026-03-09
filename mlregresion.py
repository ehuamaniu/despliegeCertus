import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('diabetes.csv')

# Predeciremos Glucose
X = data.drop(["Glucose", "Outcome"], axis=1)
y = data["Glucose"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

modelo = DecisionTreeRegressor()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("Error cuadrático medio (MSE):", mse)
print("R2 del modelo:", r2)

print("\nIngrese los datos del paciente para predecir su nivel de glucosa")

preg = float(input("Número de embarazos: "))
bp = float(input("Presión arterial: "))
skin = float(input("Grosor de piel: "))
insulin = float(input("Nivel de insulina: "))
bmi = float(input("Indice de masa corporal (BMI): "))
pedigree = float(input("Factor genético diabetes: "))
age = float(input("Edad: "))

paciente = pd.DataFrame([[preg, bp, skin, insulin, bmi, pedigree, age]],
                        columns=X.columns)

resultado = modelo.predict(paciente)

print("\nNivel de glucosa predicho:", resultado[0])