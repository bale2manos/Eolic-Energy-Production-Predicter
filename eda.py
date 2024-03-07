from collections import Counter
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns  # visualisation
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV,GridSearchCV
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt  # visualisation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

sns.set(color_codes=True)

file_path = './wind_ava.csv'
df = pd.read_csv(file_path)

"""GOOGLE COLAB
from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)
file_path = '/content/gdrive/MyDrive/wind_ava.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
"""

num_instances, num_features = df.shape
print("Number of instances: ", num_instances)
print("Number of features: ", num_features)
variable_types = df.dtypes
columns = df.columns
missing_values = df.isnull().sum()
print("Tipos de variables:")
for i in range(len(variable_types)):
    if variable_types[i] in ['float64', 'int64']:
        print(columns[i], ": ", variable_types[i], "-> numérico. || Missing values: ", missing_values[i])
    else:
        print(columns[i], ": ", variable_types[i], "-> categórico. || Missing values: ", missing_values[i])


# Mostrar columnas con valores faltantes
print("Columnas con valores faltantes:")
if missing_values.sum() == 0:
    print("No hay columnas con valores faltantes")
else:
    for col in columns:
        if missing_values[col] > 0:
            print(col, ": ", missing_values[col])

# Columnas constantes
constant_columns = df.columns[df.nunique() == 1]
print("Columnas constantes:")
if len(constant_columns) == 0:
    print("No hay columnas constantes")
else:
    for col in constant_columns:
        print(col)

# Filas duplicadas
duplicated_rows = df.duplicated()
print("Filas duplicadas: ", duplicated_rows.sum())

# Filas vacías
empty_rows = df.isnull().all(axis=1)
print("Filas vacías: ", empty_rows.sum())

# Naturaleza del problema (regresión o clasificación)
problem_type = "Problema de Regresión" if df['energy'].dtype in ['float64', 'int64'] else "Problema de Clasificación"
print(f"Naturaleza del problema: {problem_type}")

# Eliminación de variables meteorológicas no correspondientes a la localización 13
print("Columnas relevantes:")
relevant_columns = [col for col in df.columns if col.endswith(".13") or col in ['datetime', 'energy']]
df_relevant = df[relevant_columns]
print("Número de columnas relevantes: ", len(df_relevant.columns))

#Ahora calcularemos cual es la media de energía, el valor mínimo y el valor máximo:
#Encontramos esta información relevante para evaluar como de bien van a estar los modelos observando el mrse y para ello el promedio
#nos dará una pista de si ese mrse es bueno o no:
maximo = df_relevant['energy'].max()
minimo = df_relevant['energy'].min()
promedio = df_relevant['energy'].mean()

print("Máximo de la columna 'energía':", maximo)
print("Mínimo de la columna 'energía':", minimo)
print("Promedio de la columna 'energía':", promedio)

#Nos da un máximo de 279255 y un mínimo de 1.
#Por otro lado, el promedio es

# Renombrado de columnas
# Renombrar las columnas relevantes de manera más corta
df_relevant = df_relevant.rename(columns={
    't2m.13': 'temp_2m',
    'u10.13': 'wind_U_10m',
    'v10.13': 'wind_V_10m',
    'u100.13': 'wind_U_100m',
    'v100.13': 'wind_V_100m',
    'cape.13': 'convective_energy',
    'flsr.13': 'log_surface_roughness_heat_forecast',
    'fsr.13': 'surface_roughness_forecast',
    'iews.13': 'eastward_turbulent_stress',
    'inss.13': 'northward_turbulent_stress',
    'lai_hv.13': 'lai_high_vegetation',
    'lai_lv.13': 'lai_low_vegetation',
    'u10n.13': 'neutral_wind_10m_U',
    'v10n.13': 'neutral_wind_10m_V',
    'stl1.13': 'soil_temp_level_1',
    'stl2.13': 'soil_temp_level_2',
    'stl3.13': 'soil_temp_level_3',
    'stl4.13': 'soil_temp_level_4',
    'sp.13': 'surface_pressure',
    'p54.162.13': 'vertical_integral_temp',
    'p59.162.13': 'vertical_integral_div_kinetic_energy',
    'p55.162.13': 'vertical_integral_water_vapour',
    'datetime': 'datetime',
    'energy': 'energy'
})

#sns.relplot(data=df_relevant, x='surface_pressure', y='energy')
sns.regplot(x="surface_pressure", y="energy", data=df_relevant)
plt.show()
print("··································")

print(df_relevant.head(6))

df_relevant['datetime'] = pd.to_datetime(df_relevant['datetime'])

# Extraer componentes relevantes
df_relevant['year'] = df_relevant['datetime'].dt.year
df_relevant['month'] = df_relevant['datetime'].dt.month
df_relevant['day'] = df_relevant['datetime'].dt.day
df_relevant['hour'] = df_relevant['datetime'].dt.hour

#TODO Esto está bien?
# Eliminar la columna original 'datetime'
df_relevant = df_relevant.drop(columns=['datetime'])


# TODO


"""
2. Decidir cómo se va a llevar a cabo la evaluación outer (estimación de rendimiento futuro 
/ evaluación de modelo) y la evaluación inner (para comparar diferentes alternativas y ajustar hiper
parámetros). Decidir qué métrica(s) se van a usar. Justificar las decisiones.

Debido a que nos encontramos frente a una serie temporal, según los consejos del profesorado
Hemos decidido usar TimeSeriesSplit para llevar a cabo tanto la outer como la inner evaluation.
Para el outer se usarán 5 splits y para el inner 3.
La evaluación outer se utiliza para estimar el rendimiento futuro del modelo en datos no vistos, 
mientras que la inner se enfoca en el ajuste de hiperparámetros y la comparación de modelos.

La métrica que se va a usar es el rmse ya que nos parece la métrica más fácil de interpretar ya que está en las mismas unidades que la variable objetivo.

"""


"""
3.Decidir, usando KNN el método de escalado más apropiado para este problema y usarlo 
de aquí en adelante cuando sea necesario. 

Para este ejercicio, probaremos con 3 métodos de escalado (standard, Minmax y Robust) y escogeremos el que tenga el rmse más bajo:
Para evitar tener suerte a la hora de splitear los datos, usaremos un cross validation que usará un inner_cv el cual es un 
time series split con n = 3 para que al meterle al cross validation X e y, nos aseguremos tanto de que no vamos a evaluar algo en el pasado con 
algo del futuro y además de no tener suerte al dividir en train y test

Un problema puede ser el cómputo utilizado ya que la validación cruzada es más costosa
TODO esto está bien explicado?

La mejor alternativa sobre qué método de escalado es mejor usar, la decidiremos observando las medias que salen de cada uno de los cross validation para 
cada pipeline.

"""
inner_cv = TimeSeriesSplit(n_splits=3)

X,y = df_relevant.drop(columns=['energy']),df_relevant['energy']

scores = {}
#Min max
pipeline_min_max = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])

scores_min_max = cross_val_score(pipeline_min_max,X,y,cv = inner_cv,scoring="neg_root_mean_squared_error")
scores["MinMaxScaler"] = -np.mean(scores_min_max)

#Standard
pipeline_standard = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', neighbors.KNeighborsRegressor())
])

scores_std = cross_val_score(pipeline_standard,X,y,cv = inner_cv,scoring="neg_root_mean_squared_error")
scores["StandardScaler"] = -np.mean(scores_std)

#Robust
pipeline_robust = Pipeline([
    ('scaler', RobustScaler()),
    ('knn', neighbors.KNeighborsRegressor())
])

scores_robust = cross_val_score(pipeline_robust,X,y,cv = inner_cv,scoring="neg_root_mean_squared_error")
scores["RobustScaler"] = -np.mean(scores_robust)


print("Para MinMaxScaler, la media de rmse es: ",scores["MinMaxScaler"])
print("Para Standard, la media de rmse es: ",scores["StandardScaler"])
print("Para Robust, la media de rmse es: ",scores["RobustScaler"])







"""
Como podemos observar en el print:
Para MinMaxScaler, la media de rmse es:  545.1412378487257
Para Standard, la media de rmse es:  479.37296174076965
Para Robust, la media de rmse es:  461.2175957436966
La media más baja para el rmse es la del Robust scaler por lo que usaremos ese escalador cuando usemos KNN regressor.
"""

#TODO Qué diferencia hay entre cross validation y random o grid search CV

"""
Para nosotros inner Cross validation es sacar la media del scoring elegido.
Grid o random search además de la media, te da la mejor config de hiperparámetros.
"""

"""
4. A continuación, se considerarán estos métodos: KNN, árboles de regresión, regresión 
lineal (la normal y al menos, la variante Lasso) y SVM: 
a. Se evaluarán dichos modelos con sus hiperparámetros por omisión. También se medirán los 
tiempos que tarda el entrenamiento. 
b. Después, se ajustarán los hiperparámetros más importantes de cada método y se obtendrá 
su evaluación. Medir tiempos del entrenamiento, ahora con HPO. --> ELEGIR UNO ENTRE GRID, ¡¡¡¡RANDOM!!!!! Y BAYESIAN
c. 
Obtener algunas conclusiones, tales como: ¿cuál es el mejor método? ¿Cuál de los métodos 
básicos de aprendizaje automático es más rápido? ¿Los resultados son mejores que los 
regresores triviales/naive/dummy --> media aritmetica / movil ? 
¿El ajuste de hiperparámetros mejora con respecto a los 
valores por omisión? ¿Hay algún equilibrio entre tiempo de ejecución y mejora de 
resultados? ¿Es posible extraer de alguna técnica qué atributos son más relevantes? etc. 

"""




"""
4.a
Para evaluar los modelos con hpo, haremos un cross validation con time series split de 3 con X e y y calcularemos la media del rmse de cada uno.
Para ello, crearemos una pipeline con cada método (KNN, decision tree regresor...)
De todas estas medias, las evaluamos y escogemos el modelo cuya media sea la mejor.
Todo esto con el escalador Robust.
"""





"""KNN por omisión"""
X, y = df_relevant.drop(columns=['energy']), df_relevant['energy']
inner_cv = TimeSeriesSplit(n_splits=3)

# KNN con RobustScaler
pipeline_KNN = Pipeline([
    ('scaler', RobustScaler()),
    ('knn', neighbors.KNeighborsRegressor())
])
scores_KNN_hpo = cross_val_score(pipeline_KNN, X, y, cv=inner_cv, scoring="neg_root_mean_squared_error")
score_KNN = -np.mean(scores_KNN_hpo)

print("Estimación de rendimiento de KNN con RobustScaler:", score_KNN)

# Decision tree por omisión
pipeline_Dec_tree = Pipeline([
    ('decision_tree', DecisionTreeRegressor())
])
scores_Dec_tree_hpo = cross_val_score(pipeline_Dec_tree, X, y, cv=inner_cv, scoring="neg_root_mean_squared_error")
score_Dec_tree = -np.mean(scores_Dec_tree_hpo)

print("Estimación de rendimiento de Decision tree:", score_Dec_tree)

# MODELO FINAL
#modelo_final = pipeline_Robust.fit(X,y)



"""
4.b 
Hacemos un outer loop y dentro usamos gridsearch con un param_grid ya definido y un inner también con 
timeseries split para obtener el modelo con los mejores hiperparámetros. 
Del modelo que salga de outer fold, sacamos el rmse de ese outer fold negative_mse = regr.best_score_
luego hacemos una media de las (k outer folds) y con esa media evaluamos todos los modelos con
ajuste de hiperparámetros
"""

#KNN sin omisión


#Decision tree sin omisión
inicio = time.time()

outer_cv = TimeSeriesSplit(n_splits=5)
inner_cv = TimeSeriesSplit(n_splits=3)

param_grid = {'max_depth': [2, 4, 6, 8, 10, 12, 14],
 'min_samples_split': [2, 4, 6, 8, 10, 12, 14]}

regr = GridSearchCV(DecisionTreeRegressor(random_state=1),
                         param_grid,
                         scoring='neg_root_mean_squared_error',
                         # 3-fold for hyper-parameter tuning
                         cv=inner_cv,
                         n_jobs=1, verbose=1,
                        )

scores = -cross_val_score(regr,
                            X, y,
                            scoring='neg_root_mean_squared_error',
                            cv = outer_cv)
media = np.mean(scores)
print("Con optimización de hiperparámetros, para Decision tree la media del mrse es: ",media)

fin = time.time()
tiempo_transcurrido = fin - inicio
print("Tiempo transcurrido en DECISION TREE regressor con config de hiperparámetros:", tiempo_transcurrido, "segundos")
