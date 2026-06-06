# 🔍 Detección de Fraude en Tarjetas de Crédito

### Convenio PLUS TI — Universidad del Valle 2025

Implementación de métricas custom para reducción de falsos positivos (Objetivo A) y entrenamiento de un modelo federado cross-banco (Objetivo B), sobre datasets sintéticos de transacciones de tarjetas de crédito en formato ISO 8583.

---

## 📁 Estructura del Repositorio

```
/
├── .gitignore
├── informe detección de fraudes.pdf       # Informe técnico final (≤ 5 páginas)
├── 100%inferencias/                       # Predicciones is_fraud sobre Banco 3
│   ├── inferencia_100pct_v2_fedavg_recalibrado.xlsx
│   ├── inferencia_100pct_v3_fedavg_ponderado.xlsx
│   ├── inferencia_30pct_v2_fedavg_recalibrado_corrected.xlsx
│   ├── inferencia_30pct_v3_fedavg_ponderado.xlsx
│   └── inferencia_30pct_v3_fedavg_ponderado_corrected.xlsx
├── README_objetivo_a.md                   # Documentación detallada Objetivo A
├── README_objetivo_b.md                   # Documentación detallada Objetivo B
├── modelo_federado.ipynb                  # Entrenamiento y evaluación Objetivo B
├── objetivoA_FraudDetection.ipynb         # 7 modelos con métricas custom (Objetivo A)
└── Analisis Exploratorio/
    ├── EDA_bo_vip.ipynb                   # EDA Banco 1 — Bolivia VIP
    ├── EDA_br_privado.ipynb               # EDA Banco 2 — Brasil Privado
    ├── EDA_gt_estatal.ipynb               # EDA Banco 3 — Guatemala Estatal
    └── dataset_federado.ipynb             # Construcción del dataset federado BO + BR
```

---

## 🧭 Contexto

El proyecto trabaja sobre tres datasets de transacciones de tarjetas de crédito (estándar ISO 8583), correspondientes a bancos de Bolivia, Brasil y Guatemala. Los bancos 1 y 2 cuentan con etiqueta `is_fraud`; el Banco 3 no. El período cubre enero–junio 2025 (~100,000 transacciones por banco, 5,000 clientes VIP/privados/estatales).

El mes de **junio 2025** se utiliza como set de evaluación en todos los experimentos.

---

## 🎯 Objetivo A — Métricas Custom para Reducción de Falsos Positivos

> **Enfoque del grupo:** Maximizar el monto total de fraude recuperado en USD.

Se entrenaron **7 modelos LightGBM** sobre el dataset de Banco 1 (Bolivia VIP), organizados en dos pasos:

### Paso 4 — Reducción de FP ratio (punto común)

Threshold calibrado para garantizar **recall ≥ 90%** sobre la curva precision-recall.

| Modelo         | feval          | Recall | FP Ratio            | % Monto Salvado |
| -------------- | -------------- | ------ | ------------------- | --------------- |
| **Base** | AUC-ROC        | 90.04% | **0.9065** ✅ | 96.65%          |
| P4-M1          | feval_fp_ratio | 90.04% | 0.9132              | 96.51%          |
| P4-M2          | feval_balanced | 90.04% | 0.9185              | 96.79%          |

→ **Seleccionado: Modelo Base** (menor FP ratio manteniendo recall = 90%)

### Paso 6 — Maximizar monto salvado (objetivo del grupo)

Threshold calibrado maximizando utilidad económica = Σ(amount_usd de TP) − 5 × FP.

| Modelo          | feval / estrategia         | Recall           | FP Ratio | Monto Salvado USD    | % Monto             |
| --------------- | -------------------------- | ---------------- | -------- | -------------------- | ------------------- |
| P6-M3           | feval_high_value_precision | 82.42%           | 0.717    | $2,485,844           | 93.73%              |
| P6-M4           | feval_economic_utility     | 82.42%           | 0.699    | $2,484,320           | 93.67%              |
| P6-M5           | feval_amount_fbeta (β=2)  | 82.54%           | 0.746    | $2,489,412           | 93.86%              |
| **P6-M6** | sample_weight = √amount   | **86.69%** | 0.881    | **$2,591,238** | **97.70%** ✅ |

→ **Seleccionado: P6-M6** (mayor monto salvado; penaliza errores en fraudes de alto valor durante el gradiente)

Ver detalles completos en [`README_objetivo_a.md`](README_objetivo_a.md) y [`objetivoA_FraudDetection.ipynb`](objetivoA_FraudDetection.ipynb).

---

## 🌐 Objetivo B — Modelo Federado

Se entrenaron dos estrategias federadas sobre el dataset combinado BO + BR (meses 1–5), evaluadas en el test de junio:

| Modelo             | Estrategia                                         | AUC-ROC          | Recall           | FP Ratio        | % Monto Salvado     |
| ------------------ | -------------------------------------------------- | ---------------- | ---------------- | --------------- | ------------------- |
| Modelo A           | Dataset Unificado (1 solo LightGBM)                | 0.8719           | 90.56%           | 0.943           | 95.83%              |
| **Modelo B** | FedAvg Simulado (promedio de probabilidades 50/50) | **0.8858** | **90.00%** | **0.939** | **96.58%** ✅ |

→ **Seleccionado: Modelo B** — mejor AUC y mayor monto salvado, sin compartir datos crudos entre bancos.

Las inferencias sobre el Banco 3 (Guatemala Estatal) se encuentran en la carpeta [`100%inferencias/`](100%inferencias/), con versiones sobre el 30% preliminar y el 100% final, en variantes recalibrada (v2) y ponderada (v3).

Ver detalles en [`README_objetivo_b.md`](README_objetivo_b.md) y [`modelo_federado.ipynb`](modelo_federado.ipynb).

---

## 🔧 Ingeniería de Variables

Cuatro variables de comportamiento construidas de forma idéntica sobre los tres datasets (ventanas rodantes temporales, sin data leakage):

| Variable                     | Descripción                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `time_since_last_txn_min`  | Minutos desde la última transacción del mismo cliente. Velocidad anómala = posible fraude.                          |
| `txn_count_last_1h / 24h`  | Conteo de transacciones del cliente en la última hora y en las últimas 24h. Correlación con `is_fraud`: r = 0.41. |
| `amount_zscore_customer`   | Z-score del monto respecto al histórico propio del cliente. Alta desviación = transacción inusual.                  |
| `amount_ratio_vs_baseline` | Cociente monto / baseline histórico del cliente. Agnóstico a la moneda del banco.                                    |

---

## ⚙️ Entorno y Dependencias

```bash
python >= 3.10
lightgbm >= 4.0
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Se recomienda ejecutar en **Google Colab** con GPU (Tesla T4) para los notebooks de modelado.

---

## 📦 Datasets

Los datasets no están incluidos en el repositorio por su tamaño. Pueden descargarse desde:

🔗 [Google Drive — Dataset oficial PLUS TI](https://drive.google.com/drive/folders/1zC-LjQzIF2U4XEmXXSLOcJWDY6fPnYKc?usp=sharing)

| Archivo                          | Descripción                                |
| -------------------------------- | ------------------------------------------- |
| `01_bo_vip_seed22_n100000.csv` | Banco 1 — Bolivia VIP (con etiqueta)       |
| `02_br_privado_*.csv`          | Banco 2 — Brasil Privado (con etiqueta)    |
| `03_gt_estatal_*.csv`          | Banco 3 — Guatemala Estatal (sin etiqueta) |
