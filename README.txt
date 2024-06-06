
### README

# Predicción de la Demanda Eléctrica con Modelos de Machine Learning

Este proyecto se centra en la predicción de la demanda eléctrica utilizando modelos de machine learning. Se utilizan diferentes enfoques para crear y evaluar modelos de predicción, incluyendo un modelo de referencia (baseline) y modelos autoregresivos con y sin variables exógenas. Los modelos se entrenan y guardan para su posterior uso. Previamente presentaremos los resultados del analisis exploratorio.

### Análisis Exploratorio de Datos (EDA) - Demanda Eléctrica

#### Descripción general de los datos

- **Demand**:
  - Media: 4665.63 MWh
  - Desviación estándar: 871.19 MWh
  - Mínimo: 2864.29 MWh
  - Máximo: 9313.05 MWh

- **Temperature**:
  - Media: 16.26 °C
  - Desviación estándar: 5.65 °C
  - Mínimo: 1.60 °C
  - Máximo: 43.10 °C

- **Holiday**:
  - Media: 0.03 (indica que hay pocos días festivos en los datos)

#### Distribución de la demanda por hora del día

- La demanda es más baja durante las horas de la madrugada (0-6 am) y aumenta gradualmente durante el día, alcanzando su punto máximo en las horas de la tarde y noche (5-10 pm).
- Mayor variabilidad en la demanda durante las horas pico, lo que sugiere fluctuaciones más grandes en el consumo eléctrico.

#### Distribución de la demanda por mes del año

- La demanda es más alta en los meses de invierno y verano (picos alrededor de enero y julio-agosto), lo cual puede estar relacionado con el uso de calefacción y aire acondicionado respectivamente.
- Mayor variabilidad de la demanda en estos meses, indicando fluctuaciones más grandes en el consumo eléctrico durante las estaciones extremas.

#### Relación entre temperatura y demanda eléctrica

- A temperaturas más bajas y más altas, la demanda tiende a ser mayor, lo que puede estar relacionado con el uso de calefacción y aire acondicionado.
- A temperaturas moderadas (alrededor de 15-20 °C), la demanda tiende a ser más baja.

#### Matriz de correlación

- **Demand y Temperature**: Correlación positiva (a medida que aumenta la temperatura, también lo hace la demanda eléctrica, probablemente debido al uso de aire acondicionado).
- **Demand y Holiday**: Correlación negativa (la demanda tiende a ser más baja en los días festivos).

#### Serie temporal de la demanda eléctrica

- La serie temporal muestra tendencias y estacionalidad evidentes en la demanda eléctrica.
- Patrones estacionales con picos recurrentes que podrían estar relacionados con los cambios estacionales en el clima y el comportamiento humano.




## Estructura del Código

### Importación de Librerías

```python
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.datasets import fetch_dataset
```

Se importan las librerías necesarias para el procesamiento de datos, el modelado y la gestión de archivos.

### Directorio de Guardado

```python
save_dir = 'C:\Users\arzua\OneDrive\Escritorio\preelec\src\'
```

Se define el directorio donde se guardarán y cargarán los modelos entrenados.

### Carga y Preparación de Datos

```python
datos = fetch_dataset(name='vic_electricity', raw=True)
datos['Time'] = pd.to_datetime(datos['Time'], format='%Y-%m-%dT%H:%M:%SZ')
datos = datos.set_index('Time')
datos = datos.asfreq('30min')
datos = datos.sort_index()
datos = datos.drop(columns='Date')
datos = datos.resample(rule='H', closed='left', label='right').mean()
datos = datos.loc['2012-01-01 00:00:00': '2014-12-30 23:00:00'].copy()
```

1. Se cargan los datos de demanda eléctrica.
2. Se convierte la columna `Time` a formato datetime y se establece como índice.
3. Se ajusta la frecuencia de los datos a intervalos de 30 minutos.
4. Se elimina la columna `Date` y se re-muestrean los datos a intervalos de una hora.
5. Se selecciona un rango específico de fechas para el análisis.

### División de Datos

```python
fin_train = '2013-12-31 23:59:00'
fin_validacion = '2014-11-30 23:59:00'
datos_train = datos.loc[:fin_train, :].copy()
datos_val = datos.loc[fin_train:fin_validacion, :].copy()
datos_test = datos.loc[fin_validacion:, :].copy()
```

Se dividen los datos en conjuntos de entrenamiento, validación y prueba.

### Creación y Entrenamiento de Modelos

#### Modelo Baseline

```python
forecaster_baseline = ForecasterEquivalentDate(
    offset=pd.DateOffset(days=1),
    n_offsets=1
)
forecaster_baseline.fit(y=datos.loc[:fin_validacion, 'Demand'])
```

Se crea y entrena un modelo de referencia que utiliza el valor de la demanda del mismo día en el año anterior.

#### Modelo Autoregresivo Recursivo

```python
forecaster_autoreg = ForecasterAutoreg(
    regressor=LGBMRegressor(random_state=15926, verbose=-1),
    lags=24
)
forecaster_autoreg.fit(y=datos.loc[:fin_validacion, 'Demand'])
```

Se crea y entrena un modelo autoregresivo recursivo utilizando el regresor `LGBMRegressor`.

#### Modelo Autoregresivo Recursivo con Variables Exógenas

```python
exog_features = ['Temperature', 'Holiday']
forecaster_autoreg_exog = ForecasterAutoreg(
    regressor=LGBMRegressor(random_state=15926, verbose=-1),
    lags=24
)
forecaster_autoreg_exog.fit(
    y=datos.loc[:fin_validacion, 'Demand'],
    exog=datos.loc[:fin_validacion, exog_features]
)
```

Se crea y entrena un modelo autoregresivo recursivo que incorpora variables exógenas (temperatura y festivos).

### Guardado de Modelos

```python
joblib.dump(forecaster_baseline, save_dir + 'forecaster_baseline.pkl')
joblib.dump(forecaster_autoreg, save_dir + 'forecaster_autoreg.pkl')
joblib.dump(forecaster_autoreg_exog, save_dir + 'forecaster_autoreg_exog.pkl')
```

Se guardan los modelos entrenados en el directorio especificado.

### Carga de Modelos

```python
forecaster_baseline_loaded = joblib.load(save_dir + 'forecaster_baseline.pkl')
forecaster_autoreg_loaded = joblib.load(save_dir + 'forecaster_autoreg.pkl')
forecaster_autoreg_exog_loaded = joblib.load(save_dir + 'forecaster_autoreg_exog.pkl')
```

Se cargan los modelos guardados para su uso posterior.

### Verificación de Modelos

```python
print(forecaster_baseline_loaded)
print(forecaster_autoreg_loaded)
print(forecaster_autoreg_exog_loaded)
```

Se imprime la información de los modelos cargados para verificar que se han cargado correctamente.

## Requisitos

- Python 3.x
- pandas
- numpy
- lightgbm
- skforecast
- joblib

## Ejecución

1. Asegúrate de tener instaladas todas las librerías necesarias.
2. Ejecuta el script para cargar, preparar los datos, entrenar los modelos y guardarlos.
3. Puedes cargar y utilizar los modelos guardados para hacer predicciones futuras.

### Notas

- Asegúrate de que la ruta `C:\Users\arzua\OneDrive\Escritorio\preelec\src\` existe y es accesible.
- Personaliza las variables exógenas según tus necesidades específicas del problema. En este ejemplo, se utilizaron `Temperature` y `Holiday` como variables exógenas.

Este README proporciona una visión general de las acciones realizadas en el script, explicando cada paso del proceso de carga de datos, preparación, modelado, guardado y carga de los modelos.
