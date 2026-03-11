%%writefile README.md
# Análisis de Predicción de Cancelación de Clientes (Churn) - TelecomX

## Introducción
Este proyecto tiene como objetivo predecir la cancelación de clientes (Churn) en la empresa de telecomunicaciones TelecomX. La cancelación de clientes es un problema crítico que afecta los ingresos y el crecimiento de las empresas. Mediante el uso de diversas técnicas de análisis de datos y modelado predictivo, buscamos identificar los factores clave que influyen en que un cliente decida cancelar su servicio y proponer estrategias para mejorar la retención.

## 1. Carga y Exploración de Datos
Los datos fueron cargados directamente desde un archivo JSON alojado en GitHub, utilizando la librería `requests` para acceder a la API y `pandas` para convertir la respuesta en un DataFrame. Esto permitió un acceso directo y eficiente a la fuente de datos.

## 2. Preprocesamiento y Transformación de Datos
La estructura original de los datos contenía varias columnas con información anidada (diccionarios). Para facilitar el análisis, se realizaron los siguientes pasos:
- **Aplanamiento de Columnas Anidadas**: Las columnas `account`, `internet`, `phone` y `customer` fueron aplanadas, extrayendo sus subcampos en nuevas columnas del DataFrame principal. Por ejemplo, de `account` se extrajeron `MonthlyCharges` y `TotalCharges`, de `internet` servicios como `OnlineSecurity` y `StreamingTV`, de `phone` servicios como `PhoneService` y `MultipleLines`, y de `customer` atributos como `gender`, `SeniorCitizen`, `Partner`, `Dependents` y `tenure`.
- **Tratamiento de Valores Nulos y Conversión de Tipos**: Las columnas `MonthlyCharges` y `TotalCharges` fueron convertidas a tipo numérico. Se identificaron 11 valores nulos en `TotalCharges` (que surgieron de clientes con `tenure` 0, es decir, nuevos clientes sin cargos totales aún), los cuales fueron eliminados del DataFrame para mantener la integridad de los datos.
- **Eliminación de `customerID`**: La columna `customerID` se eliminó, ya que es un identificador único sin valor predictivo para el modelo.

## 3. Codificación de Variables
Para preparar los datos para los modelos de aprendizaje automático, se realizó la siguiente codificación:
- **Variables Binarias**: Columnas como `Churn`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` y `gender` (Male/Female) fueron mapeadas a valores numéricos (1 para 'Yes'/'Male', 0 para 'No'/'Female'). Se manejaron casos como 'No internet service' o 'No phone service' mapeándolos a 'No'.
- **Variables Multicategóricas (One-Hot Encoding)**: Las columnas `Contract`, `PaymentMethod` e `InternetService` (que tienen más de dos categorías) fueron transformadas usando One-Hot Encoding para evitar la interpretación de una relación ordinal que no existe. Se aplicó `drop_first=True` para evitar la multicolinealidad.

## 4. Manejo del Desbalance de Clases
Se detectó un desbalance significativo en la variable objetivo `Churn` (aproximadamente 74% 'No Churn' vs. 26% 'Churn'). Para mitigar este problema y evitar que los modelos se sesgaran hacia la clase mayoritaria, se aplicó la técnica de sobremuestreo **SMOTE (Synthetic Minority Over-sampling Technique)**. Esto generó ejemplos sintéticos para la clase minoritaria ('Churn'=1), resultando en una distribución de clases balanceada (50%-50%) en el conjunto de entrenamiento.

## 5. Estandarización de Datos
Después del balanceo de clases, todas las características numéricas del DataFrame fueron estandarizadas utilizando `StandardScaler`. Este paso es crucial para algoritmos basados en distancia (como KNN y SVM) y para algunos métodos de regularización en modelos lineales (como Regresión Logística), ya que asegura que todas las características contribuyan equitativamente al modelo y evita que las características con mayores rangos dominen el cálculo.

## 6. Análisis de Correlaciones
Se calculó y visualizó una matriz de correlación para entender las relaciones entre las características y con la variable objetivo `Churn`. Se observó que `tenure`, `Contract_Two year`, `InternetService_Fiber optic` y `MonthlyCharges` mostraron las correlaciones más fuertes con `Churn`.

## 7. Modelado y Análisis de Relevancia de Variables
Se entrenaron y analizaron cuatro modelos de clasificación para predecir la cancelación, extrayendo insights sobre la importancia de las características.

### Regresión Logística
- **Entrenamiento**: Se entrenó un modelo de Regresión Logística utilizando los datos estandarizados y balanceados. Es un modelo lineal que estima la probabilidad de que un cliente cancele.
- **Insights Clave (Coeficientes)**: Los coeficientes positivos indican que un aumento en la característica correspondiente incrementa la probabilidad de Churn, mientras que los negativos la disminuyen. Las variables con los coeficientes de mayor magnitud (absoluta) son las más influyentes.
  - **`MonthlyCharges` (positivo)**: A mayores cargos mensuales, mayor probabilidad de Churn.
  - **`InternetService_Fiber optic` (positivo)**: Los clientes con fibra óptica tienen mayor probabilidad de Churn (comparado con la referencia).
  - **`tenure` (negativo)**: Una mayor antigüedad reduce significativamente la probabilidad de Churn.
  - **`Contract_Two year` y `Contract_One year` (negativos)**: Los contratos a largo plazo reducen la probabilidad de Churn.
  - **Servicios Adicionales (negativos)**: `OnlineSecurity`, `TechSupport`, `OnlineBackup`, `DeviceProtection` (cuando están presentes) reducen la probabilidad de Churn.

### K-Nearest Neighbors (KNN)
- **Entrenamiento**: Se entrenó un clasificador KNN. Este modelo clasifica un nuevo punto de datos basándose en la mayoría de sus 'k' vecinos más cercanos.
- **Insights Clave (Impacto de Escala y Dimensionalidad)**:
  - La **estandarización** fue fundamental para KNN, ya que al ser un algoritmo basado en distancia, asegura que características con rangos muy diferentes (`MonthlyCharges`, `TotalCharges`, `tenure`) no dominen el cálculo de la distancia, permitiendo que todas las características contribuyan equitativamente.
  - La **dimensionalidad** (número de características) puede ser un desafío. En espacios de alta dimensión, la noción de 'proximidad' se diluye, lo que puede afectar el rendimiento de KNN. No hay una importancia de características explícita como en otros modelos, pero el peso de cada característica se manifiesta a través de su contribución a la distancia en el espacio multidimensional.

### Random Forest
- **Entrenamiento**: Se entrenó un clasificador Random Forest, un modelo de conjunto basado en árboles de decisión.
- **Insights Clave (Importancia de las Características)**: Random Forest calcula la importancia de las características basándose en la reducción promedio de la impureza (Gini o entropía) que cada característica aporta en los árboles del bosque. Las variables con mayor importancia son las más predictivas.
  - **`MonthlyCharges`, `TotalCharges`, `tenure`**: Consistentemente las tres características más importantes, indicando su fuerte influencia en la decisión de Churn.
  - **`Contract_Two year`, `Contract_One year`**: También muy relevantes, reafirmando que los tipos de contrato son factores clave de retención.
  - **`OnlineSecurity`, `TechSupport`, `InternetService_Fiber optic`**: Estos servicios también destacan como importantes, con la fibra óptica y la ausencia de seguridad/soporte técnico siendo predictores clave de Churn.

### Support Vector Machine (SVM) Lineal
- **Entrenamiento**: Se entrenó un clasificador SVM con un kernel lineal. Este modelo busca el hiperplano óptimo que separa las clases en el espacio de características.
- **Insights Clave (Coeficientes y Frontera de Decisión)**:
  - Para SVMs lineales, la interpretación es similar a la Regresión Logística: los coeficientes (`model.coef_`) indican la dirección y magnitud del impacto de cada característica en la frontera de decisión. Mayores magnitudes significan mayor influencia.
  - Se observaron patrones similares a la Regresión Logística: `MonthlyCharges`, `InternetService_Fiber optic`, `tenure`, `Contract_Two year` y `Contract_One year`, junto con los servicios adicionales como `OnlineSecurity` y `TechSupport`, mostraron los coeficientes más altos en magnitud, confirmando su importancia en la separación de clientes que cancelan de los que no.
  - A diferencia de los kernels no lineales, donde la interpretación directa es compleja, el kernel lineal permite entender la contribución de cada variable a la separación de clases.

## 8. Conclusiones Clave y Estrategias de Retención
El análisis de los diferentes modelos ha revelado un conjunto consistente de factores altamente influyentes en la predicción de la cancelación de clientes:

**Factores Clave de Influencia en la Cancelación:**
1.  **Cargos Mensuales (`MonthlyCharges`)**: Los clientes con cargos mensuales más altos son significativamente más propensos a cancelar.
2.  **Antigüedad del Cliente (`tenure`)**: Los clientes de mayor antigüedad son mucho menos propensos a cancelar.
3.  **Tipo de Contrato (`Contract_Two year`, `Contract_One year`)**: Los contratos a largo plazo son un fuerte indicador de retención; los clientes con contratos de mes a mes tienen una mayor tasa de cancelación.
4.  **Servicio de Internet (`InternetService_Fiber optic`, `InternetService_No`)**: Los clientes con servicio de fibra óptica y aquellos sin ningún servicio de internet (indicando la ausencia de internet por parte de la compañía) son factores importantes. La fibra óptica, a pesar de ser un servicio 'premium', mostró un coeficiente positivo en modelos lineales, lo que podría indicar que las expectativas o problemas asociados a este servicio pueden llevar a la cancelación.
5.  **Servicios Adicionales (`OnlineSecurity`, `TechSupport`, `OnlineBackup`, `DeviceProtection`)**: La ausencia o falta de contratación de estos servicios de valor añadido aumenta la probabilidad de Churn. Su presencia actúa como un factor de retención.
6.  **Cargos Totales (`TotalCharges`)**: Aunque correlacionado con `MonthlyCharges` y `tenure`, también es un predictor importante.

**Estrategias de Retención Propuestas:**
-   **Revisión de Precios y Ofertas para Cargos Mensuales Altos**: Implementar programas de fidelización o descuentos dirigidos a clientes con altos cargos mensuales que muestren signos de insatisfacción o que se encuentren en riesgo de Churn, especialmente aquellos en contratos de mes a mes.
-   **Incentivos para Contratos a Largo Plazo**: Fomentar la migración de clientes de contratos de mes a mes a contratos de uno o dos años, ofreciendo beneficios exclusivos (descuentos, mejoras de servicio, etc.) para aumentar su compromiso y reducir la tasa de Churn.
-   **Promoción de Servicios Adicionales**: Aumentar la visibilidad y el valor percibido de servicios como seguridad online, soporte técnico, copia de seguridad y protección de dispositivos. Los clientes que utilizan estos servicios muestran una menor propensión a cancelar.
-   **Mejora de la Experiencia del Cliente de Fibra Óptica**: Investigar las razones detrás de la mayor tasa de Churn entre clientes de fibra óptica. Podría estar relacionado con expectativas no cumplidas, problemas de estabilidad, o atención al cliente. Mejorar la calidad del servicio y la comunicación es crucial.
-   **Programas de Lealtad para Clientes Recién Incorporados**: Poner especial atención en los clientes nuevos (baja `tenure`) para asegurar una experiencia positiva inicial y convertirlos en clientes a largo plazo. Pequeños incentivos o seguimiento proactivo pueden ser beneficiosos.

Este análisis proporciona una base sólida para entender el comportamiento de Churn y desarrollar estrategias proactivas para retener a los clientes de TelecomX.
