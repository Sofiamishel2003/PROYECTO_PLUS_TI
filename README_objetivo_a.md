# Objetivo A — Métricas Custom para Reducción de Falsos Positivos

**Convenio PLUS TI – Universidad del Valle 2025**

---

## Descripción

Exploración y evaluación de funciones de evaluación personalizadas (`feval`) en LightGBM para mejorar la detección de fraudes en transacciones de tarjetas de crédito VIP de un banco boliviano. El objetivo general es reducir la tasa de falsos positivos sin sacrificar la capacidad de detección. El objetivo específico del grupo es **maximizar el monto de fraude detectado** manteniendo un recall mínimo del 90%.

---

## Dataset

| Atributo            | Detalle                |
| ------------------- | ---------------------- |
| Archivo             | `bo_vip_clean.csv`   |
| Registros           | 100,003 transacciones  |
| Período            | Enero – Junio 2025    |
| Clientes            | 5,000 (segmento VIP)   |
| Tasa de fraude      | ~4.92%                 |
| Features originales | 67 → 43 tras limpieza |
| Formato             | ISO 8583               |

**Split temporal:**

- **Train:** trimestre 1 (enero–marzo) — 50,027 registros
- **Test:** trimestre 2 (abril–junio) — 49,882 registros

---

## Ingeniería de Features

Se crearon 5 features adicionales sobre los 4 del EDA original:

| Feature                       | Descripción                                         |
| ----------------------------- | ---------------------------------------------------- |
| `time_since_last_txn_min`   | Minutos desde la última transacción del cliente    |
| `txn_count_last_1h`         | Cantidad de transacciones en la última hora         |
| `txn_count_last_24h`        | Cantidad de transacciones en las últimas 24h        |
| `amount_zscore_customer`    | Z-score del monto respecto al histórico del cliente |
| `amount_ratio_vs_baseline`  | Monto actual / promedio histórico del cliente       |
| `is_night`                  | Flag horario nocturno (11pm–5am)                    |
| `international_high_amount` | Internacional + monto > percentil 90                 |
| `amount_x_zscore`           | Intensidad relativa: zscore × monto                 |
| `velocity_x_amount`         | Velocidad × monto combinados                        |

---

## Manejo del Desbalance de Clases

Se aplicó **undersampling de transacciones legítimas** a ratio 10:1 (en lugar del desbalance original de ~19:1), reduciendo el sesgo del modelo hacia predecir fraude en transacciones ambiguas. El conjunto de test no fue modificado para mantener la distribución real de producción.

```
Train balanceado: 27,819 registros (9.09% fraude)
scale_pos_weight: 10.0
```

---

## Modelos Entrenados

Se entrenaron 6 modelos LightGBM con `metric='None'` para que cada `feval` controle el early stopping de forma independiente.
| Modelo      | feval              | Descripción                             |
| ----------- | ------------------ | ---------------------------------------- |
| Modelo Base | `feval_auc`      | AUC-ROC como baseline            |

### Paso 4 — Objetivo general: reducir FP ratio

| Modelo      | feval              | Descripción                             |
| ----------- | ------------------ | ---------------------------------------- |
| P4-M1       | `feval_fp_ratio` | Minimiza `FP / (TP + FP)` directamente |
| P4-M2       | `feval_balanced` | Maximiza `recall - 0.5 × fp_ratio`    |

### Paso 6 — Objetivo del grupo: maximizar monto salvado

| Modelo | feval                         | Descripción                                                |
| ------ | ----------------------------- | ----------------------------------------------------------- |
| P6-M3  | `feval_amount_recall_floor` | Monto salvado con penalización cuadrática si recall < 90% |
| P6-M4  | `feval_amount_fp_weighted`  | `(monto_TP / monto_total) × (1 - fp_ratio)`              |
| P6-M5  | `feval_amount_fbeta`        | F-beta ponderado por monto con β=2                         |

---

## Resultados

| Modelo                                    | AUC-ROC          | Recall           | FP Ratio         | FP               | % Monto Salvado  |
| ----------------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Modelo Base — AUC/F1                     | 0.9056           | 90.04%           | 0.9102           | 21,813           | 96.37%           |
| P4-M1 — FP Ratio                         | 0.9034           | 90.04%           | 0.9158           | 23,384           | 96.45%           |
| P4-M2 — Balanced                         | 0.9047           | 90.04%           | 0.9149           | 23,124           | 96.34%           |
| P6-M3 — Amount + Recall Floor            | 0.9044           | 90.04%           | 0.9136           | 22,749           | 96.35%           |
| **P6-M4 — Amount × (1-FP Ratio)** | **0.9061** | **90.04%** | **0.9099** | **21,714** | 96.34%           |
| P6-M5 — Amount F-beta (β=2)             | 0.9045           | 90.04%           | 0.9146           | 23,051           | **96.55%** |

---

## Conclusión

Se seleccionan dos modelos según prioridad:

**P6-M4 — `feval_amount_fp_weighted`** es el modelo seleccionado como mejor balance global: logra el **menor FP ratio de todos los modelos** (0.9099), incluso por debajo del baseline, mientras mantiene recall del 90% y recupera el 96.34% del monto total de fraudes. Su diseño fuerza al modelo a equilibrar explícitamente recuperación de dinero y reducción de falsas alarmas simultáneamente.

**P6-M5 — `feval_amount_fbeta`** es la mejor opción si la prioridad es recuperación económica: alcanza el **mayor % de monto salvado (96.55%)** a costa de más FP (23,051). Útil cuando el banco prioriza el valor recuperado sobre la carga operativa del equipo de revisión.

> Las diferencias entre modelos son pequeñas porque el dataset tiene un techo de separabilidad reflejado en el AUC ~0.90. El FP ratio elevado (~0.91) en todos los modelos es una consecuencia estructural del desbalance de clases al operar con recall ≥ 90%: para no perder el 90% de los fraudes el threshold debe ser bajo, lo que arrastra inevitablemente miles de transacciones legítimas.

---

## Stack

- Python 3.10 / Google Colab (GPU Tesla T4)
- LightGBM 4.6.0
- scikit-learn, pandas, numpy, matplotlib, seaborn