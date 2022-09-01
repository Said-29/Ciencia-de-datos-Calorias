import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone

datos_consumo = pd.read_excel('datos.xlsx')
datos_consumo.head()
datos_consumo.groupby("Momento").count()
datos_consumo.describe()
datos_seleccionados = datos_consumo.iloc[:,3:8]
datos_seleccionados.info()
datos_seleccionados.isnull().values.any()
dataset = datos_seleccionados.dropna()
dataset.isnull().sum()
x = dataset[['Carbohidratos (g)', 'Lípidos/grasas (g)', 'Proteína (g)', 'Sodio (mg)']].values
y = dataset['Calorías (kcal)'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
modelo_regresion = LinearRegression()
modelo_regresion.fit(x_train, y_train)
x_columns = ['Carbohidratos (g)', 'Lípidos/grasas (g)', 'Proteína (g)', 'Sodio (mg)']
coeff_df = pd.DataFrame(modelo_regresion.coef_, x_columns, columns = ['Coeficientes'])
y_pred = modelo_regresion.predict(x_test)
validacion = pd.DataFrame({'Actual':y_test, 'Predicción':y_pred, 'Diferencia':y_test-y_pred})
muestra_validacion = validacion.head(25)
validacion["Diferencia"].describe()
print(r2_score(y_test, y_pred))
muestra_validacion.plot.bar(rot=0)
plt.title("Comparación de calorías actuales y de predicción")
plt.xlabel("Muestra de alimentos")
plt.ylabel("Cantidad de calorías")
plt.show()
zona_horaria = timezone('America/Monterrey')
fecha_hora = datetime.now(zona_horaria)
fecha_hora_formato = fecha_hora.strftime("%B %d, %Y %H:%M:%S")
print("Created on", fecha_hora_formato)
autor = "Jordan Barba"
print("Autor:", autor)