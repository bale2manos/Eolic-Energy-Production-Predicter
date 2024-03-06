from collections import Counter
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns  # visualisation
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV
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

print(df_relevant.head(6))

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
Para la outer evaluation usaremos el timesplit con 5 folds y para la inner evaluation usaremos timesplit con 3 folds para asegurarnos de que no hemos 
tenido suerte a la hora de dividir entre la parte de test y train en cada "time series fold" del "outer loop".
Un problema puede ser el cómputo utilizado ya que la validación cruzada es más costosa
TODO esto está bien explicado?

No usaremos la outer evaluation para tomar una decisión. La mejor alternativa sobre qué método de escalado es mejor usar, la decidiremos en el inner loop.

"""

outer_cv = TimeSeriesSplit(n_splits=5)
inner_cv = TimeSeriesSplit(n_splits=3)
inner_rmse_means = [[], [], []]
scores = {}
#Primer alternativa: Standard:

for train_index, test_index in outer_cv.split(df_relevant):
    X_train, X_test = df_relevant.drop(columns=['energy']).iloc[train_index], df_relevant.drop(columns=['energy']).iloc[
        test_index]
    y_train, y_test = df_relevant['energy'].iloc[train_index], df_relevant['energy'].iloc[test_index]

    pipeline_min_max = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])

    pipeline_standard = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])

    pipeline_robust = Pipeline([
        ('scaler', RobustScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])

    inner_scores1 = cross_val_score(pipeline_min_max, X_train, y_train, cv=inner_cv, scoring="neg_root_mean_squared_error")
    inner_scores1_mean = -inner_scores1.mean()
    inner_rmse_means[0].append(inner_scores1_mean)

    inner_scores2 = cross_val_score(pipeline_standard, X_train, y_train, cv=inner_cv, scoring="neg_root_mean_squared_error")
    inner_scores2_mean = -inner_scores2.mean()
    inner_rmse_means[1].append(inner_scores2_mean)

    inner_scores3 = cross_val_score(pipeline_robust, X_train, y_train, cv=inner_cv, scoring="neg_root_mean_squared_error")
    inner_scores3_mean = -inner_scores3.mean()
    inner_rmse_means[2].append(inner_scores3_mean)


scores["MinMaxScaler"] = np.mean(inner_rmse_means[0])
scores["StandardScaler"] = np.mean(inner_rmse_means[1])
scores["RobustScaler"] = np.mean(inner_rmse_means[2])
print("Para MinMaxScaler, la media de rmse es: ",scores["MinMaxScaler"])
print("Para Standard, la media de rmse es: ",scores["StandardScaler"])
print("Para Robust, la media de rmse es: ",scores["RobustScaler"])







"""
Como podemos observar en el print:
Para MinMaxScaler, la media de rmse es:  553.7883988206586
Para Standard, la media de rmse es:  487.38296471912935
Para Robust, la media de rmse es:  487.43129008467366
La media más baja para el rmse es la del Standard scaler por lo que usaremos ese escalador cuando usemos KNN regressor.
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
Usamos un for loop externo. Dentro, creamos una pipeline como modelo por cada uno 
de los métodos (KNN, decision tree regresor...). De estos, hacemos una "inner" cross validation
para sacar una media de cada uno de los outer folds y luego hacemos una media de 5 de cada media que ha salido
de los outer folds.
De todas estas medias, las evaluamos y escogemos el modelo cuya media sea la mejor.
Todo esto con el escalador standard.

4.b 
Hacemos un outer loop y dentro usamos gridsearch con un param_grid ya definido y un inner también con 
timeseries split para obtener el modelo con los mejores hiperparámetros. 
Del modelo que salga de outer fold, sacamos el rmse de ese outer fold negative_mse = regr.best_score_
luego hacemos una media de las (k outer folds) y con esa media evaluamos todos los modelos con
ajuste de hiperparámetros
"""




"""KNN por omisión"""
X,y = df_relevant.drop(columns=['energy']),df_relevant['energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=6) # TODO TimeSeries??
pipeline_standard = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])
pipeline_standard.fit(X_train, y_train)
y_pred = pipeline_standard.predict(X_test)
outer_score = metrics.mean_squared_error(y_test, y_pred)
print("Estimación de rendimiento de KNN con RobustScaler", outer_score)

# MODELO FINAL
modelo_final = pipeline_standard.fit(X,y)
































#class sklearn.model_selection.TimeSeriesSplit(n_splits=5, *, max_train_size=None, test_size=None, gap=0)

inicio = time.time()

cv_outer = TimeSeriesSplit(n_splits=5)

scores = []

# TODO ahora mismo solo es outer?
for train_index, test_index in cv_outer.split(df_relevant):
    X_train, X_test = df_relevant.drop(columns=['energy']).iloc[train_index], df_relevant.drop(columns=['energy']).iloc[test_index]
    y_train, y_test = df_relevant['energy'].iloc[train_index], df_relevant['energy'].iloc[test_index]

    """param_grid = {'knn__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                  'knn__metric': ['euclidean', 'manhattan', 'minkowski']}"""

    pipe = Pipeline([
        ('scaler', StandardScaler()), # TODO why MinMaxScaler?
        ('knn', neighbors.KNeighborsRegressor())
    ])

    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    rmse_knn = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    scores.append(rmse_knn)

score = np.array(scores)
print("All RMSE: ", score)
print("Mean accuracy: ", score.mean(),"+/-", score.std())

fin = time.time()
tiempo_transcurrido = fin - inicio
print("Tiempo transcurrido en KNN regressor:", tiempo_transcurrido, "segundos")

"DECISION TREE REGRESSOR"
inicio = time.time()
scores = []
cv_outer = TimeSeriesSplit(n_splits=5)
for train_index, test_index in cv_outer.split(df_relevant):
    X_train, X_test = df_relevant.drop(columns=['energy']).iloc[train_index], df_relevant.drop(columns=['energy']).iloc[test_index]
    y_train, y_test = df_relevant['energy'].iloc[train_index], df_relevant['energy'].iloc[test_index]

    """param_grid = {'knn__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                  'knn__metric': ['euclidean', 'manhattan', 'minkowski']}"""

    pipe = Pipeline([
        ('decision_tree', DecisionTreeRegressor())]
    )

    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    rmse_knn = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    #Calcular el error cuadrático medio (RMSE) es correcto para un problema de regresión.

    scores.append(rmse_knn)

score = np.array(scores)
print("All DECISION TREE RMSE: ", score)
print("Mean accuracy: ", score.mean(),"+/-", score.std())

fin = time.time()
tiempo_transcurrido = fin - inicio
print("Tiempo transcurrido en DECISION TREE regressor:", tiempo_transcurrido, "segundos")

"Cross validation"
X,y = df_relevant.drop(columns=['energy']),df_relevant['energy']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42)
inner = KFold(n_splits=3,shuffle=True,random_state=42)

pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', neighbors.KNeighborsRegressor())
    ])
scores = cross_val_score(pipe,X_train,y_train,cv = inner,scoring = 'neg_root_mean_squared_error')
print("Con cross validation",-scores)