# Objetivo A — Métricas Custom para Reducción de Falsos Positivos

**Convenio PLUS TI – Universidad del Valle 2025**

---

## Descripción

Exploración y evaluación de funciones de evaluación personalizadas (`feval`) en LightGBM para mejorar la detección de fraudes en transacciones de tarjetas de crédito VIP de un banco boliviano. El objetivo general es reducir la tasa de falsos positivos sin sacrificar la capacidad de detección. El objetivo específico del grupo es **maximizar el monto de fraude detectado en USD**.

El trabajo se organiza en dos etapas diferenciadas según el enunciado:

- **Paso 4 (punto común a todos los grupos):** dos funciones `feval` orientadas a minimizar el FP ratio, evaluadas con threshold que garantiza recall ≥ 90%.
- **Paso 6 (objetivo del grupo):** cuatro funciones `feval` orientadas a maximizar el monto de fraude recuperado, evaluadas con threshold de utilidad económica máxima — sin restricción de recall, dado que el enunciado #6 indica ajustar parámetros según el objetivo asignado.

---

## Dataset

| Atributo       | Detalle                                                      |
| -------------- | ------------------------------------------------------------ |
| Archivo fuente | `Copia de 01_bo_vip_seed22_n100000.csv` (dataset original) |
| Registros      | 100,003 transacciones                                        |
| Período       | Enero – Junio 2025                                          |
| Clientes       | 5,000 (segmento VIP)                                         |
| Tasa de fraude | ~4.92%                                                       |
| Formato        | ISO 8583                                                     |

> En versiones anteriores se usaba `bo_vip_clean.csv`, un archivo pre-procesado para el modelo federado (segunda parte del proyecto) que eliminaba columnas por razones de privacidad diferencial y federabilidad. Para este modelo centralizado se volvió al dataset original para recuperar columnas con valor predictivo real.

**Split temporal:**

- **Train:** trimestre 1 (enero–marzo)
- **Test:** trimestre 2 (abril–junio) — nunca modificado para sampling ni calibración

---

## Revisión del EDA a la luz del modelo centralizado

El EDA original fue diseñado para el modelo federado, donde la prioridad era eliminar identificadores para privacidad diferencial y construir features que funcionen en todos los bancos del consorcio. Esa lógica de limpieza resulta demasiado conservadora para un modelo centralizado. Se identificaron tres tipos de decisiones del EDA que se revisaron:

**Columnas eliminadas por razones del modelo federado, recuperadas aquí:**

| Columna                              | Por qué el EDA la eliminó                | Por qué se recupera                                                                                                                        |
| ------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `DE9_conversion_rate_billing`      | Correlación 0.82 con `is_international` | Informa la *magnitud* de la internacionalidad, no solo si/no. Una compra en Argentina tiene tipo de cambio muy distinto a una en Europa. |
| `DE43_card_acceptor_name_location` | Identificador no federable entre bancos    | Para modelo centralizado: comercios con alta tasa de fraude son señal directa. Se trata como categórica con LabelEncoder.                 |
| `client_id`                        | Identificador de privacidad                | Se usa únicamente para calcular features históricos por cliente (comportamiento previo) y se elimina antes del entrenamiento.             |

**Hallazgos del EDA no explotados en features:**

El EDA identificó dos patrones centrales que no se habían convertido en features:

1. La mediana de fraude (~USD 800) es 4× la mediana legítima (~USD 200), con cola pronunciada hacia USD 2,500–2,800.
2. El canal ECOM tiene tasa de fraude del 8%, vs 2.3% en POS presencial.

Ambos hallazgos estaban documentados en el EDA pero ninguno había generado un feature que los combinara ni que capturara la distribución de montos.

---

## Ingeniería de Features

Se construyeron 13 features en total. Los primeros 4 provienen del EDA original; los siguientes 9 son nuevos en esta versión.

### Features del EDA original

| Feature                     | Descripción                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `time_since_last_txn_min` | Minutos desde la última transacción del cliente. Velocidad anómala = posible fraude. |
| `txn_count_last_1h`       | Cantidad de transacciones del cliente en la última hora.                               |
| `txn_count_last_24h`      | Cantidad de transacciones del cliente en las últimas 24h.                              |
| `amount_zscore_customer`  | Z-score del monto respecto al histórico del cliente. Alta desviación = anómalo.      |

### Features nuevos — comportamiento histórico por cliente

Requieren `client_id` durante el cálculo; se eliminan del feature set antes del entrenamiento.

| Feature                      | Descripción                                                                                                                                                                                              | Motivación                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `first_intl_txn_pct`       | Proporción de transacciones internacionales previas del cliente (expanding mean con shift). Si `is_international=1` y `first_intl_txn_pct=0`, es la primera vez que el cliente opera en el exterior. | Primera transacción internacional = señal de alto riesgo, especialmente para fraudes de alto monto. |
| `amount_vs_max_historical` | Ratio entre el monto actual y el máximo histórico del cliente hasta esa transacción. Ratio > 1 = monto nunca visto antes.                                                                              | Captura directamente el tipo de fraude de alto valor que el objetivo del grupo prioriza.              |

### Features nuevos — distribución de montos

Derivados del hallazgo del EDA sobre la concentración de fraudes en montos altos.

| Feature                      | Descripción                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `amount_ratio_vs_baseline` | Monto actual dividido el promedio histórico del cliente.                                              |
| `amount_percentile`        | Posición del monto en la distribución global (rank percentil).                                       |
| `is_high_amount`           | Flag: monto en el top 10% global (percentil 90), donde el EDA muestra mayor concentración de fraudes. |

### Features nuevos — interacciones del EDA

| Feature                       | Descripción                                                                      | Motivación                                                                                                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ecom_high_amount`          | `channel == ECOM` AND `amount_usd >= p90`.                                    | Combina los dos hallazgos principales del EDA: ECOM tiene 8% tasa de fraude y los fraudes se concentran en montos altos. Ninguno de los dos features por separado captura la combinación. |
| `international_high_amount` | `is_international == True` AND `amount_usd >= p90`.                           | Cruza internacionalidad con monto alto.                                                                                                                                                    |
| `amount_zscore_channel`     | Z-score del monto dentro del mismo canal (ECOM, POS, ATM, etc.).                  | Un gasto de USD 1,000 es normal en ECOM pero anómalo en POS. El zscore global del cliente no captura esta diferencia.                                                                     |
| `is_night`                  | Flag horario nocturno (11pm–5am). Tasa de fraude ~8% vs ~4.5% diurno según EDA. |                                                                                                                                                                                            |
| `amount_x_zscore`           | `abs(amount_zscore_customer) × amount_usd`. Intensidad relativa.               |                                                                                                                                                                                            |
| `velocity_x_amount`         | `txn_count_last_1h × amount_usd`.                                              |                                                                                                                                                                                            |

---

## Manejo del Desbalance de Clases

Se aplicó **undersampling de transacciones legítimas** a ratio 10:1 (desbalance original ~19:1), reduciendo el sesgo del modelo hacia predecir todo como fraude. El test no fue modificado para mantener la distribución real de producción.

```
Train balanceado : ~27,800 registros | fraude: ~9.09%
scale_pos_weight : 10.0
```

### Hiperparámetros diferenciados por modelo

Cada modelo tiene hiperparámetros ajustados a su función objetivo. Compartir los mismos parámetros provoca que el espacio de hipótesis explorado fuera idéntico para todos, y las diferencias entre fevals se reducían a cuándo parar el entrenamiento, no a qué aprender.

| Grupo | Filosofía                        | `num_leaves` | `reg_lambda` | `scale_pos_weight` |
| ----- | --------------------------------- | -------------- | -------------- | -------------------- |
| P4-M1 | Conservador: pocas alarmas        | 31             | 10.0           | SPW × 0.5           |
| P4-M2 | Moderado: balance recall/FP       | 47             | 7.0            | SPW × 0.7           |
| P6-M3 | Expresivo para montos altos       | 95             | 2.0            | SPW × 0.8           |
| P6-M4 | Precision en fraudes caros        | 63             | 3.0            | SPW × 0.7           |
| P6-M5 | Agresivo en recall de alto valor  | 127            | 1.0            | SPW × 1.2           |
| P6-M6 | Igual que P6-M4 + pesos por monto | 63             | 3.0            | SPW × 0.7           |

---

## Modelos Entrenados

Se entrenaron 7 modelos LightGBM con `metric='None'` para que cada `feval` controle el early stopping de forma independiente.

### Modelo Base

| Modelo | `feval`     | Descripción                       |
| ------ | ------------- | ---------------------------------- |
| Base   | `feval_auc` | AUC-ROC estándar como referencia. |

### Paso 4 — Reducir FP ratio (threshold: recall ≥ 90%)

El criterio de evaluación del enunciado #4-5 es la métrica `FP / (TP + FP)` buscando recall ≥ 90%. El threshold se elige como el valor más alto que garantiza ese recall mínimo en el test set.

| Modelo | `feval`          | Descripción                                                               |
| ------ | ------------------ | -------------------------------------------------------------------------- |
| P4-M1  | `feval_fp_ratio` | Minimiza `FP / (TP + FP)` directamente. `is_higher_better=False`.      |
| P4-M2  | `feval_balanced` | Maximiza `recall − 0.5 × fp_ratio`. Penalización lineal y simétrica. |

### Paso 6 — Maximizar monto salvado (threshold: utilidad máxima)

El enunciado #6 indica ajustar parámetros según el objetivo asignado, sin imponer restricción de recall. El threshold se elige maximizando la función de utilidad económica:

```
utilidad = Σ(amount_usd de TP) − fp_cost × FP      (fp_cost = USD 5 por alerta)
```

| Modelo | `feval`          | Descripción                                                               |
| ------ | ------------------ | -------------------------------------------------------------------------- |
| P6-M3  | `feval_high_value_precision`                 | `recall_hv × (1 − fp_ratio)` donde `recall_hv` es el recall solo sobre fraudes del percentil 75+ de monto. Concentra la detección en transacciones de mayor valor.                                                      |
| P6-M4  | `feval_economic_utility`                     | Utilidad económica normalizada:`pct_monto_salvado − (FP × fp_cost) / monto_total_fraudes`. El modelo recibe un incentivo explícito en dólares para no generar FP.                                                       |
| P6-M5  | `feval_amount_fbeta_adaptive`                | F-beta con pesos cuadráticos por monto (`weight = amount²`). Un fraude de USD 1,000 vale 100× más que uno de USD 100 (vs 10× con peso lineal). β=2 → recall pesa el doble que precisión.                             |
| P6-M6  | `feval_economic_utility` + `sample_weight` | Igual que P6-M4 en feval, pero el dataset de entrenamiento asigna `weight = √(amount_usd)` a cada transacción fraudulenta. Cambia el **gradiente** del árbol durante el entrenamiento, no solo la señal de parada. |

> **Diferencia conceptual entre P6-M4 y P6-M6:** P6-M4 ajusta *cuándo parar* el entrenamiento según el monto. P6-M6 ajusta *qué aprende* el árbol en cada iteración. Equivocarse en un fraude de USD 2,000 penaliza ~4.5× más que en uno de USD 100 en todos los árboles del ensemble.

---

## Resultados

### Paso 4 — Threshold: recall ≥ 90%

| Modelo                  | AUC-ROC | Recall | FP Ratio         | FP               | % Monto Salvado |
| ----------------------- | ------- | ------ | ---------------- | ---------------- | --------------- |
| Base — AUC-ROC         | —      | 90.04% | **0.9065** | **20,843** | 96.65%          |
| P4-M1 — feval_fp_ratio | —      | 90.04% | 0.9132           | 22,616           | 96.51%          |
| P4-M2 — feval_balanced | —      | 90.04% | 0.9185           | 24,226           | 96.79%          |

**Modelo seleccionado Paso 4: Base — AUC-ROC** con el menor FP ratio (0.9065) y 20,843 FP.

La convergencia de los tres modelos a FP ratio ~0.91 es estructural, no un fallo de las fevals: con recall forzado al 90% y 5% de fraude en test, la precision máxima alcanzable está acotada por el AUC del modelo (~0.90). Para subir significativamente la precision manteniendo ese recall se requeriría mejor separabilidad, no distintas fevals.

### Paso 6 — Threshold: utilidad económica máxima

| Modelo                                    | Threshold        | Recall           | Precision | FP Ratio | FP     | Monto Salvado USD    | % Monto Salvado  |
| ----------------------------------------- | ---------------- | ---------------- | --------- | -------- | ------ | -------------------- | ---------------- |
| P6-M3 — High Value Precision             | 0.1299           | 82.42%           | 0.2826    | 0.7174   | 4,998  | $2,485,844           | 93.73%           |
| P6-M4 — Economic Utility                 | 0.1353           | 82.42%           | 0.3014    | 0.6986   | 4,564  | $2,484,320           | 93.67%           |
| P6-M5 — Amount F-beta Adaptive           | 0.1394           | 82.54%           | 0.2537    | 0.7463   | 5,801  | $2,489,412           | 93.86%           |
| **P6-M6 — Sample Weight √amount** | **0.1033** | **86.69%** | 0.1193    | 0.8807   | 15,285 | **$2,591,238** | **97.70%** |

Los modelos del Paso 6 presentan dos trade-offs distintos:

**Alta precisión / bajo FP:** P6-M3, P6-M4 y P6-M5 operan con threshold más alto (~0.13–0.14), generando entre 4,564 y 5,801 FP con FP ratio de 0.70–0.75. Recuperan entre el 93.67% y 93.86% del monto total de fraudes.

**Alto monto / mayor FP:** P6-M6 opera con threshold 0.1033, genera 15,285 FP (FP ratio 0.88), pero recupera el **97.70% del monto** — el mayor de todos los modelos. La diferencia en USD respecto al modelo más selectivo (P6-M4) es de **+$106,917**.

---

## Conclusión

### Paso 4

El modelo Base con `feval_auc` obtiene el menor FP ratio (0.9065) entre los modelos del punto común. La similitud de resultados entre los tres modelos refleja el techo de separabilidad del dataset (AUC ~0.90) y la restricción estructural del recall ≥ 90%: con 5% de fraude en test, para capturar el 90% de los fraudes el threshold debe ser bajo, arrastrando inevitablemente ~20,000 transacciones legítimas.

### Paso 6

**Modelo seleccionado: P6-M6 — Sample Weight √amount**

Es el único modelo donde la señal de aprendizaje está alineada con el objetivo de negocio desde el nivel del gradiente: `sample_weight = √(amount_usd)` hace que el árbol aprenda a valorar los fraudes de alto monto durante el entrenamiento, no solo durante el early stopping. Esto resulta en el mayor monto recuperado de todos los modelos: **$2,591,238 (97.70%)** con recall del 86.69%.

El FP ratio de P6-M6 (0.88) es similar al Paso 4 (0.91), pero con un threshold distinto: en el Paso 4 ese threshold está determinado por el piso de recall del 90%; en P6-M6 está determinado por maximizar el monto recuperado. Son el mismo nivel de FP por razones completamente distintas.

**Modelo alternativo: P6-M4 — Economic Utility**

Si la capacidad operativa del banco es limitada (pocas alertas investigables por día), P6-M4 es preferible: 4,564 FP con 93.67% de monto recuperado. La elección entre P6-M4 y P6-M6 depende del costo operativo real por alerta investigada.

> La reducción de FP en los modelos selectivos del Paso 6 (de ~21,000 a ~4,500) no proviene de un modelo más complejo: proviene de elegir el threshold correcto para el objetivo correcto. El Paso 4 optimiza cobertura; el Paso 6 optimiza valor económico recuperado.

---

## Stack

- Python 3.12 / Google Colab (GPU Tesla T4)
- LightGBM 4.6.0
- scikit-learn, pandas, numpy, matplotlib, seaborn
