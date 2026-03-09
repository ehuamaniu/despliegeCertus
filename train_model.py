import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

# Leer datos
data = pd.read_csv("diabetes.csv")

# Entradas y salida
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Crear y entrenar modelo
modelo = DecisionTreeClassifier(random_state=2)
modelo.fit(X, y)

# Guardar modelo entrenado
with open("modelo_diabetes.pkl", "wb") as archivo:
    pickle.dump(modelo, archivo)

print("Modelo guardado correctamente")