import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score

data = pd.read_csv('diabetes.csv')


#rint(data.head())

X = data.drop("Outcome", axis=1)

y = data["Outcome"]

print("Caracterisitca de los pacientes\n",X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

modelo = DecisionTreeClassifier()

modelo.fit(X_train, y_train)


predicciones = modelo.predict(X_test)

accuracy = accuracy_score(y_test, predicciones)

print("Precision del modelo " , accuracy)


print("Ingrese los datos del paciente")

preg = float(input("Número de embarazos: "))
glucose = float(input("Nivel de glucosa: "))
bp = float(input("Presión arterial: "))
skin = float(input("Grosor de piel: "))
insulin = float(input("Nivel de insulina: "))
bmi = float(input("Indice de masa corporal (BMI): "))
pedigree = float(input("Factor genético diabetes: "))
age = float(input("Edad: "))



paciente = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]],
                        columns=X.columns)

resultado = modelo.predict(paciente)


if resultado[0] == 1 :
    print("El paciente tiene diabetes")
else:
    print("El paciente no tiene diabetes")





