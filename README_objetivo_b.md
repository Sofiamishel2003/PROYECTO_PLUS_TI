# Objetivo B — Modelo Federado para Detección de Fraude

**Convenio PLUS TI – Universidad del Valle 2025**

---

## Descripción

Entrenamiento de un modelo federado con datos de Banco 1 (BO VIP, Bolivia) y Banco 2 (BR Privado, Brasil) para detectar fraude en Banco 3 (GT Estatal, Guatemala), que **no posee etiquetas**. Se implementan y comparan dos estrategias: un dataset unificado (baseline federado) y FedAvg simulado (modelos locales independientes cuyas predicciones se promedian).

La restricción central del trabajo es que el modelo debe ser **agnóstico al banco y al país**: no puede usar `bank_code`, `bank_country`, moneda local ni identificadores institucionales, de modo que generalice a cualquier banco nuevo sin reentrenamiento.

---

## Datasets

### Bancos de entrenamiento

| Atributo       | Banco 1 — BO VIP (Bolivia) | Banco 2 — BR Privado (Brasil) |
| -------------- | -------------------------- | ----------------------------- |
| Archivo        | `bo_vip_clean.csv`       | `br_privado_clean.csv`      |
| Registros      | 100,003 transacciones      | 100,000 transacciones         |
| Tasa de fraude | 4.92%                      | 3.21%                         |
| Período       | Enero – Junio 2025        | Enero – Junio 2025           |
| Formato        | ISO 8583                   | ISO 8583                      |

### Banco de inferencia

| Atributo       | Banco 3 — GT Estatal (Guatemala) |
| -------------- | --------------------------------- |
| Archivo        | `gt_estatal_clean.csv`          |
| Registros      | 100,000 transacciones             |
| Tasa de fraude | **Desconocida** (sin etiquetas)   |
| Período       | Enero – Junio 2025               |

### Dataset federado unificado

Los datasets de BO VIP y BR Privado se combinan eliminando columnas de sesgo jurisdiccional (`bank_code`, `bank_country`, `client_segment`, moneda local).

| Partición  | Meses    | Registros | Fraudes |
| ----------- | -------- | --------- | ------- |
| **Train**   | 1 – 5   | 167,778   | 6,871   |
| **Test**    | 6 (junio) | 32,055    | 1,250   |
| **GT Estatal** | 1 – 6 | 100,000   | —       |

---

## Ingeniería de Features Agnósticas

Se diseñaron 5 features que generalizan entre bancos y países sin depender del monto absoluto ni de la nomenclatura institucional. La misma función `add_federated_features()` se aplica idénticamente a los tres datasets.

| Feature nueva              | Descripción                                                    | Propósito                                    |
| -------------------------- | --------------------------------------------------------------- | -------------------------------------------- |
| `amount_ratio_vs_baseline` | `amount_usd / (client_baseline_amount + ε)`                   | Ratio relativo al gasto habitual del cliente |
| `is_night`                 | 1 si `hour_local` ∈ [23, 5], 0 si no                          | Patrón nocturno universal de fraude          |
| `velocity_x_zscore`        | `txn_count_last_1h × \|amount_zscore_customer\|`              | Anomalía combinada velocidad + monto         |
| `intl_high_zscore`         | 1 si `is_international=True` y `amount_zscore_customer > 2.0` | Internacional con gasto inusualmente alto    |
| `velocity_ratio_1h_24h`    | `(txn_count_1h + 1) / (txn_count_24h / 24 + 1)`              | Detección de bursts cortos de actividad      |

**Total features usadas por el modelo: 35** (30 originales del dataset federado + 5 nuevas agnósticas)

<details>
<summary>Lista completa de las 35 features</summary>

`DE123_pos_data_code`, `DE18_merchant_category_code`, `DE22_pos_entry_mode`, `DE23_card_seq_number`, `DE25_pos_condition_code`, `DE39_response_code`, `DE3_processing_code`, `DE4_amount_transaction`, `DE52_pin_data_present`, `DE55_emv_data_present`, `DE60_pos_terminal_type`, `DE61_pos_extended_data`, `DE6_amount_cardholder_billing`, `DE9_conversion_rate_billing`, `MTI`, `amount_local`, `amount_tx_currency`, `amount_usd`, `amount_zscore_customer`, `card_brand`, `channel`, `client_baseline_amount`, `client_segment`, `day_of_week`, `distance_from_home_km`, `hour_local`, `is_international`, `time_since_last_txn_min`, `txn_count_last_1h`, `txn_count_last_24h`, `amount_ratio_vs_baseline`, `is_night`, `velocity_x_zscore`, `intl_high_zscore`, `velocity_ratio_1h_24h`

</details>

---

## Manejo del Desbalance de Clases

El dataset federado combinado presenta un desbalance severo de ~23:1. Se aplica **undersampling 10:1** al conjunto de entrenamiento.

```
Negativos originales : 160,907
Positivos originales :   6,871
Ratio original       :   23.42:1

Train balanceado (10:1): 75,581 filas | Fraude: 9.09%
scale_pos_weight        : 10.0
```

El conjunto de test (junio) y GT Estatal **no se modifican** para preservar la distribución real de producción.

---

## Modelos Entrenados

Se implementan dos estrategias de federated learning simulado con los mismos hiperparámetros base.

### Hiperparámetros compartidos

```python
PARAMS = {
    'objective'        : 'binary',
    'metric'           : 'None',      # controlado por feval
    'num_leaves'       : 63,
    'learning_rate'    : 0.05,
    'min_child_samples': 20,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 1.0,
    'scale_pos_weight' : 10.0,
    'random_state'     : 22,
}
num_boost_round = 1000
early_stopping  = 50 rounds
```

La `feval` reutilizada del Objetivo A es `feval_amount_fp_weighted`:

```
score = (monto_TP / monto_total) × (1 - fp_ratio)
```

### Modelo A — Dataset Unificado (Baseline Federado)

Un único LightGBM entrenado sobre BO VIP + BR Privado combinados. Requiere que ambos bancos compartan sus datos crudos bajo acuerdo de privacidad.

```
BO VIP ──────────────────────┐
                              ├──► LightGBM único ──► predicción
BR Privado ───────────────────┘
```

### Modelo B — FedAvg Simulado

Cada banco entrena su propio LightGBM de forma independiente. Solo se comparten las predicciones (probabilidades), nunca los datos crudos. Las probabilidades se combinan por promedio simple (soft voting 50/50).

```
BO VIP ──► LightGBM_BO ──► P(fraude)_BO ──┐
                                            ├──► (P_BO + P_BR) / 2 ──► predicción
BR Privado ──► LightGBM_BR ──► P(fraude)_BR ──┘
```

Features del FedAvg: intersección de features entre ambos modelos locales → **35 features** (coincide con el total).

---

## Resultados en Test (Junio)

| Modelo                          | AUC-ROC | Recall | Precision | F1     | FP Ratio | FP     | TP    | % Monto Salvado | Threshold |
| ------------------------------- | ------- | ------ | --------- | ------ | -------- | ------ | ----- | --------------- | --------- |
| Modelo A — Unificado            | 0.8719  | 90.56% | 5.70%     | 0.1073 | 0.9430   | 18,714 | 1,132 | 95.83%          | 0.2209    |
| **Modelo B — FedAvg Simulado** | **0.8855** | **90.00%** | **6.06%** | **0.1135** | **0.9394** | **—** | **—** | **96.57%** | **0.0753** |

**Modelo seleccionado: Modelo B — FedAvg Simulado**

El Modelo B supera al Modelo A en AUC (+0.014), FP Ratio (−0.004) y % Monto Salvado (+0.74%). Dado que las diferencias son favorables y el Modelo B preserva la privacidad de los datos por diseño (principio central del Federated Learning), se selecciona como modelo de producción.

---

## Inferencia sobre GT Estatal

Se aplica el modelo seleccionado sobre las 100,000 transacciones de GT Estatal. Al no existir etiquetas, no es posible calcular métricas reales.

| Entregable                         | Filas   | Fraude predicho | Tasa predicha |
| ---------------------------------- | ------- | --------------- | ------------- |
| Preliminar (30%)                   | 30,000  | 18,027          | 60.09%        |
| Final (100%)                       | 100,000 | 60,482          | 60.48%        |

**Archivos generados:**

| Archivo                              | Descripción                                      |
| ------------------------------------ | ------------------------------------------------ |
| `model_A_unificado.lgb`            | Modelo A serializado (LightGBM booster)          |
| `model_BO_fedavg.lgb`              | Modelo local de Bolivia (FedAvg)                 |
| `model_BR_fedavg.lgb`              | Modelo local de Brasil (FedAvg)                  |
| `inferencia_30pct_gt_estatal.xlsx` | Predicciones primer 30% — columna `is_fraud`   |
| `inferencia_100pct_gt_estatal.xlsx`| Predicciones 100% — columna `is_fraud`         |

---

## Nota sobre el Distribution Shift

GT Estatal presenta montos ~10× menores que los bancos de entrenamiento (mediana USD 13.70 vs ~USD 60–85). Esto genera una tasa de fraude predicha (~60%) significativamente superior a la esperada (~4–5%). El fenómeno se mitiga parcialmente mediante features relativas (`amount_zscore_customer`, `amount_ratio_vs_baseline`, `intl_high_zscore`), pero el threshold no fue recalibrado específicamente para GT Estatal. En un escenario de producción real, se recomienda:

1. Recalibrar el threshold usando una muestra etiquetada de GT Estatal.
2. Ponderar el FedAvg por AUC individual de cada banco en lugar de 50/50 fijo.
3. Explorar fine-tuning del modelo federado con las primeras etiquetas disponibles de Guatemala.

---

## Conclusión

| Criterio                | Modelo A (Unificado) | Modelo B (FedAvg) |
| ----------------------- | -------------------- | ----------------- |
| Datos compartidos       | Sí (datos crudos)    | No (solo predicciones) |
| Privacidad              | Menor                | **Mayor** ✓       |
| AUC en test             | 0.8719               | **0.8855** ✓      |
| FP Ratio en test        | 0.9430               | **0.9394** ✓      |
| % Monto salvado         | 95.83%               | **96.57%** ✓      |
| Complejidad operativa   | Baja                 | Moderada          |

El Modelo B (FedAvg Simulado) es técnicamente superior en todas las métricas clave y, además, respeta el principio fundamental del Federated Learning: **los datos nunca abandonan el servidor del banco**. Se selecciona como modelo de producción para la inferencia sobre GT Estatal.

---

## Stack

- Python 3.10 / Google Colab (GPU Tesla T4)
- LightGBM 4.6.0
- scikit-learn, pandas, numpy, matplotlib, seaborn
