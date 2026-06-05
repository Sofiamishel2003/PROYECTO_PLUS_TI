# Objetivo B — Modelo Federado para Detección de Fraude

**Convenio PLUS TI – Universidad del Valle 2025**

---

## Descripción

Entrenamiento de un modelo federado con datos de Banco 1 (BO VIP, Bolivia) y Banco 2 (BR Privado, Brasil) para detectar fraude en Banco 3 (GT Estatal, Guatemala), que **no posee etiquetas**. Se implementan y comparan dos estrategias: un dataset unificado (baseline federado) y FedAvg simulado (modelos locales independientes cuyas predicciones se promedian). Sobre Banco 3 se generan **tres inferencias federadas** (V1/V2/V3) que exploran cómo combinar los bancos y cómo recalibrar el umbral ante la ausencia de etiquetas.

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
| **Modelo B — FedAvg Simulado** | **0.8856** | **90.00%** | **6.09%** | **0.1141** | **0.9391** | **17,345** | **1,125** | **96.58%** | **0.0754** |

**Modelo seleccionado: Modelo B — FedAvg Simulado**

El Modelo B supera al Modelo A en AUC (+0.0137), FP Ratio (−0.0039) y % Monto Salvado (+0.75%). Dado que las diferencias son favorables y el Modelo B preserva la privacidad de los datos por diseño (principio central del Federated Learning), se selecciona como modelo de producción.

---

## Inferencia sobre GT Estatal — Tres variantes federadas

Sobre las 100,000 transacciones de GT Estatal se generan **tres inferencias**, todas con el **modelo federado (FedAvg)**: cada banco entrena de forma local y solo se promedian las probabilidades (los datos crudos nunca se comparten). Al no existir etiquetas, no es posible calcular métricas reales; las variantes exploran, en progresión, cómo combinar los bancos y cómo calibrar el umbral.

| Variante                              | Combinación de bancos                  | Threshold            | Tasa fraude 30% | Tasa fraude 100% |
| ------------------------------------- | --------------------------------------- | -------------------- | --------------- | ---------------- |
| **V1 — FedAvg base**                  | Promedio 50/50                          | 0.0754 (operativo)   | 42.81%          | 43.22%           |
| **V2 — FedAvg recalibrado**           | Promedio 50/50                          | 0.1171 (calibrado ~5%) | 4.89%         | 5.00%            |
| **V3 — FedAvg ponderado por AUC**     | Ponderado por AUC (0.557 / 0.443)       | 0.1092 (calibrado ~5%) | 4.88%         | 5.00%            |

- **V1** usa el umbral operativo calibrado a recall 90% en junio. Predice una tasa de fraude elevada (~43%), evidenciando el **distribution shift**: los montos de GT son ~10× menores que los de entrenamiento, lo que infla `amount_zscore_customer` y dispara la señal de fraude. Sirve como baseline.
- **V2** corrige el shift recalibrando el umbral por cuantil para marcar solo el 5% de transacciones con mayor score (prevalencia asumida similar a la regional de Bolivia/Brasil).
- **V3** mantiene esa calibración pero, en lugar de promediar 50/50, pondera cada banco por su AUC individual en junio (Bolivia ≈ 0.876 → peso 0.557; Brasil ≈ 0.697 → peso 0.443), dando más peso al modelo que generaliza mejor.

V2 y V3 coinciden en el 99.64% de las predicciones del 30%; V1 vs V2 coinciden solo en 62.07% (efecto de la recalibración). Las entregas se exportan con las columnas `transaction_id` + `is_fraud` ("Verdadero"/"Falso"), respetando el orden original del archivo de Banco 3; el 30% son las primeras 30,000 transacciones de ese archivo.

**Archivos generados:**

| Archivo                                          | Descripción                                          |
| ------------------------------------------------ | ---------------------------------------------------- |
| `model_A_unificado.lgb`                          | Modelo A serializado (LightGBM booster)              |
| `model_BO_fedavg.lgb`                            | Modelo local de Bolivia (FedAvg)                     |
| `model_BR_fedavg.lgb`                            | Modelo local de Brasil (FedAvg)                      |
| `inferencia_{30,100}pct_v1_fedavg_base.xlsx`     | V1 — promedio 50/50, umbral operativo                |
| `inferencia_{30,100}pct_v2_fedavg_recalibrado.xlsx` | V2 — promedio 50/50, umbral recalibrado ~5%       |
| `inferencia_{30,100}pct_v3_fedavg_ponderado.xlsx`| V3 — ponderado por AUC, umbral recalibrado ~5%       |

---

## Nota sobre el Distribution Shift

GT Estatal presenta montos ~10× menores que los bancos de entrenamiento (mediana USD 13.70 vs ~USD 60–85). Con el umbral operativo sin recalibrar (V1), esto genera una tasa de fraude predicha (~43%) muy superior a la esperada (~4–5%). El fenómeno se mitiga mediante features relativas (`amount_zscore_customer`, `amount_ratio_vs_baseline`, `intl_high_zscore`) y, sobre todo, mediante la **recalibración del umbral por cuantil** aplicada en V2 y V3, que lleva la tasa predicha a ~5%, consistente con la prevalencia regional de Bolivia/Brasil. Como trabajo futuro se recomienda:

1. Recalibrar el threshold con la **retroalimentación real del primer 30%** de GT Estatal remitida por PLUS TI (en lugar de asumir prevalencia ~5%).
2. La ponderación del FedAvg por AUC individual (V3) ya sustituye al 50/50 fijo; afinarla con las primeras etiquetas disponibles.
3. Explorar fine-tuning / adaptación de dominio sobre las variables de monto antes de la inferencia final.

> La selección final entre V2 y V3 se apoyará en las métricas del primer 30% devueltas por PLUS TI.

---

## Conclusión

| Criterio                | Modelo A (Unificado) | Modelo B (FedAvg) |
| ----------------------- | -------------------- | ----------------- |
| Datos compartidos       | Sí (datos crudos)    | No (solo predicciones) |
| Privacidad              | Menor                | **Mayor** ✓       |
| AUC en test             | 0.8719               | **0.8856** ✓      |
| FP Ratio en test        | 0.9430               | **0.9391** ✓      |
| % Monto salvado         | 95.83%               | **96.58%** ✓      |
| Complejidad operativa   | Baja                 | Moderada          |

El Modelo B (FedAvg Simulado) es técnicamente superior en todas las métricas clave y, además, respeta el principio fundamental del Federated Learning: **los datos nunca abandonan el servidor del banco**. Se selecciona como modelo de producción para la inferencia sobre GT Estatal.

---

## Stack

- Python 3.10 / Google Colab (GPU Tesla T4)
- LightGBM 4.6.0
- scikit-learn, pandas, numpy, matplotlib, seaborn
