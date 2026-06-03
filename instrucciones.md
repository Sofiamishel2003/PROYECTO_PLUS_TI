
# Convenio PLUS TI – Universidad del Valle 2025

## Trabajo teórico-práctico 1:

#### A) “Métricas personalizadas para reducción de falsos positivos en clasificación binaria:

#### is_fraude:”Verdadero”;”Falso”.

#### B) “Entrenamiento de Modelo Federado”.

## Objetivo A):

El objetivo de este trabajo es explorar y evaluar distintas estrategias para mejorar la detección de fraudes en
un conjunto de datos proporcionados. Actualmente, el modelo presenta una alta tasa de falsos positivos por
cada fraude identificado. Para abordar este problema, se propone probar diversas métricas personalizadas en
la función de evaluación (feval) de LightGBM, con el fin de identificar cuál logra reducir los falsos positivos sin
afectar negativamente la capacidad de detección.
Además, se busca explorar funciones de evaluación que permitan optimizar aspectos específicos del
modelo, como la precisión, la cobertura o la priorización de alertas relevantes.

1. Diseñar y evaluar funciones de evaluación personalizadas enfocadas en reducir la cantidad de
   falsos positivos por cada fraude detectado. Este será el punto de partida común para todos
   los grupos de estudiantes.
2. Asignar a cada grupo un objetivo específico de optimización, el cual servirá como enfoque
   principal para desarrollar y ajustar sus modelos.
3. Cada grupo deberá proponer 2 funciones de evaluación distintas alineadas con el objetivo
   asignado, implementarlas y compararlas entre sí para analizar cuál se desempeña mejor.
4. Seleccione la función de evaluación más efectiva entre las propuestas, justificando la elección
   con base en los resultados obtenidos.

### Descripción del Dataset

Para este trabajo se debe utilizar el conjunto de datos 01_bo_vip_seed22_n100000.csv. Este es un conjunto de datos creado
sintéticamente con transacciones de tarjetas de crédito de un banco de Bolivia de clientela VIP. El Dataset
contiene transacciones legítimas y fraudulentas desde el 1 de enero de 20 25 hasta el 3 0 de junio de 202 5.
Cubre tarjetas de crédito correspondientes a 5 000 clientes que realizan transacciones nacionales e
internacionales. Cuenta con 67 variables originales. Se usará como prueba de evaluación el mes de Junio 2025.
El formato del conjunto de datos es el ISO 8583 (estándar para tarjetas de crédito).

### Metodología

#### 1. Análisis exploratorio de datos (AED)

* Cargar el conjunto de datos en un entorno de trabajo como Python con Pandas y NumPy.
* Realizar un análisis de distribución y balanceo de clases.

#### 2. Ingeniería de variables

* Crear nuevas variables que aporten información relevante al modelo, como contadores, acumuladores,
  condiciones. Algunos ejemplos:
  * time_since_last_txn_min: Minutos desde la última transacción del mismo cliente. Velocidad
    anómala = posible fraude.
  * txn_count_last_1h: Cantidad de transacciones del cliente en la última hora.
  * txn_count_last_24h: Cantidad de transacciones del cliente en las últimas 24 horas.
  * cantidad_zscore_customer: Z-score del monto de la transacción respecto al histórico del
    cliente. Alta desviación = anómalo.

#### 3. Implementación del modelo base

* Entrenar un modelo inicial con LightGBM utilizando métricas tradicionales como AUC-
  ROC y F1-score.
* Evaluar el rendimiento inicial en términos de fraude detectado y falsos positivos.

#### 4. Definición de métricas personalizadas

* Implementar diferentes funciones feval para LightGBM, por ejemplo:
* Penalización de falsos positivos.
* Métricas balanceadas que favorecen una menor tasa de falsos positivos sin reducir la
  prevención de fraude.
* La métrica que evaluamos es la ratio de falsos positivos definida como: ratio falsos positivos = FP / (TP + FP).

#### 5. Evaluación de resultados

* Compare el rendimiento de las distintas métricas feval usando el último trimestre del conjunto de datos como prueba.
  El comparativo debería ser la proporción de falsos positivos buscando tener un 90% de detección.
* Determinar cuál estrategia logra el mejor balance entre falsos positivos y detección de fraude.

#### 6. Optimización con objetivos diferenciados por grupos de alumnos

* A cada grupo se le asignará un objetivo de optimización específica, el cual puede estar relacionado con
  distintos aspectos del desempeño del modelo. En función del objetivo asignado, cada grupo deberá diseñar al
  menos tres funciones de evaluación personalizadas.
* Ajustar parámetros del modelo y/o pesos de clases según sea necesario para cumplir con el objetivo.
* Evaluar y comparar el rendimiento de cada función diseñada, seleccionando la más efectiva y justificando su
  elección con evidencia cuantitativa obtenida del modelo.

## Objetivo B):

Se entregarán los datasets etiquetados de Banco 1 y Banco 2. El dataset de Banco 3 se entregará sin la
variable objetivo(is_fraud). Los equipos deberán entrenar un modelo federado utilizando Banco 1 y Banco 2 y
generar para cada transacción de Banco 3 una predicción binaria.

Los 3 bancos son de distintos países, con clientela y tipologías de fraude diferentes.
Esta es una práctica muy novedosa y requiere investigación.
El conjunto de datos del 3er banco no contiene la marcación del fraude. Deberán entrenar un modelo utilizando la data
del Banco 1 y 2, y realizar inferencias sobre el primer 30% de las transacciones del banco 3.
Durante la realización del trabajo práctico deberán enviarnos las inferencias del primer 30% del banco 3 para
poder devolverles las métricas sobre cómo van con este 3er dataset.
Con nuestras devoluciones podrán ir identificando qué metodología es la mejor y lo aplicarán al restante 70%
del dataset como entrega final.
El objetivo de esta segunda parte del TP es realizar la mejor inferencia sobre el 70% restante de las
transacciones del Banco 3.

### Metodología

#### 1. Análisis exploratorio de datos (AED)

* Cargar los conjuntos de datos en un entorno de trabajo como Python con Pandas y NumPy.
* Realizar un análisis de distribución y balanceo de clases.

#### 2. Ingeniería de variables

* Crear nuevas variables en los 3 conjuntos de datos que aportan información relevante al modelo, como contadores,
  acumuladores, condiciones.

#### 3. Investigar procedimientos y adaptaciones necesarias para entreamientos de Modelos

#### Federados. Entrenar modelo.

* Tenga cuidado que el modelo no debe aprender sobre cosas específicas como “banco” o lugar de “fraude”.
  Debe tener cierto agnosticismo sobre ciertas variables locales que no se aplican como variables globales.
* Es posible tener que realizar algunas transformaciones a los datos, según la estrategia que se decida llevar a
  cabo.

#### 4. Aplicar, modelo entrenado con datasets 1 y 2 en el banco 3

* Dede el sábado 9 /5 hasta el miércoles 3 /6 se deben entregar por lo menos 3 inferencias realizadas sobre el
  primer 30% de las transacciones del dataset 3, con el fin que podamos remitirles a la brevedad los resultados
  de este primer 30%, y así puedan identificar la mejor metodología.
  Las entregas preliminares de este 30% deben ser por correo, en un archivo Excel que contiene únicamente en
  la columna A, la variable “is_fraud” indicando: Verdadero o Falso.

#### 5.Aplicar la inferencia final sobre todo el dataset del banco 3 para ser evaluado y presentar

#### métricas de resultado esperado.

* El día 5 de junio deben brindar una presentación y ppt de su trabajo realizado, de una duración de 5
  minutos, y entregar el repositorio con el dataset 3 con su inferencia sobre el 100% de las transacciones.
  A partir de la investigación desarrollada y esta inferencia podremos evaluar el resultado final del Modelo
  Federado.

## Recomendaciones

* Documentar cada paso del proceso en el cuaderno.
* Utilizar visualizaciones para respaldar el análisis.
* Realizar optimización de hiperparámetros
* Explicar el razonamiento detrás de cada decisión tomada.

## Entrega

El trabajo deberá entregarse en forma de un informe técnico de no más de 5 páginas, adjuntando los
cuadernos utilizados para la implementación y los resultados de los experimentos A) y B).
El informe debe incluir:

* Resumen
* Metodología
* Descripción de la práctica de implementación
* Conclusiones
* Análisis de los resultados de la evaluación, con énfasis en el comparativo de estrategias.
* Conjunto de datos 3 con la inferencia realizada

## Recursos Adicionales

Se puede optar por utilizar Google Colab para facilitar el uso de recursos computacionales, incluyendo GPU si
es necesario. Se recomienda el uso de bibliotecas de Python como Scikit-learn, lgbm para la implementación
de los modelos.

Enlace descarga Conjunto de datos:
https://drive.google.com/drive/folders/1zC-LjQzIF2U4XEmXXSLOcJWDY6fPnYKc?usp=sharing

Contacto y mail para envíos de inferencias del 30% y consultas:

* Gabriel Andrade Castejón.
* Correo electrónico: gandrade@plusti.com
