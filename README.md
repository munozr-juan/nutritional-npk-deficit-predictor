# Early NPK Deficit Predictor: End-to-End IoT & Edge AI Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32--C3%2FDevKit-orange.svg)](https://platformio.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%2F%20Joblib-green.svg)](https://scikit-learn.org/)

Este repositorio contiene la arquitectura de hardware, firmware y el pipeline de Machine Learning desarrollado para la detección temprana de deficiencias de Macronutrientes (Nitrógeno, Fósforo y Potasio) en cultivos de ciclo corto (*Black Simpson Lettuce*). El sistema fusiona telemetría de sustrato via Modbus RTU, espectrometría visible en el borde y un pipeline predictivo multiclase.

---

## 1. Arquitectura General y Flujo de Datos

El sistema se divide en dos macro-ecosistemas acoplados de manera asíncrona: la capa de adquisición e instrumentación (IoT) y la capa de analítica predictiva (Modelado).

![Diagrama de Flujo de la Solución](docs/images/arch_flow.jpg)

### Descripción del Flujo:
1. **Capa IoT:** El microcontrolador central (ESP32) gestiona la adquisición síncrona de variables macroambientales de sustrato mediante una sonda NPK acoplada por bus industrial, y la firma espectral de las hojas mediante un sensor espectrométrico.
2. **Ingesta e Interfaz:** Los datos son estructurados localmente para almacenamiento en búfer local o transmisión directa mediante pasarelas de red hacia los sistemas de almacenamiento centralizados.
3. **Capa de Modelado:** Los datos crudos ingresan a un pipeline secuencial de ETL donde se limpian, se extraen índices de reflectancia espectral no lineales, se normalizan las escalas de características físicas y se ejecuta la inferencia estadística para clasificar el estado nutricional del cultivo.

---

## 🔌 2. Conexiones de Hardware y Esquemático Eléctrico

La etapa de potencia e instrumentación está diseñada para aislar el ruido de conmutación de la fuente industrial de las etapas de lectura digital y analógica de los sensores.

> ⚠️ **[Marcador de Posición]** El esquema eléctrico detallado en formato CAD/KiCad se encuentra actualmente en proceso de exportación final y se ubicará en esta sección.
> 
> ![Esquemático Eléctrico del Sistema](docs/images/hardware_schematic.png)

### Especificaciones de Interconexión:
* **Unidad Central de Proceso:** ESP32 DevKitC (SRAM integrada para almacenamiento intermedio de tramas).
* **Sensor Espectrométrico:** Adafruit AS7262 de 6 canales multiespectrales en el espectro visible, comunicado por bus $I^2C$ a 400 kHz.
* **Sonda de Sustrato:** Sensor NPK industrial basado en el protocolo RS485. Requiere excitación de tensión externa y aislamiento de datos.
* **Transceptor de Datos:** Módulo MAX485 para la conversión de niveles diferenciales RS485 a niveles TTL binarios directos para la UART del ESP32.

---

## 🛠️ 3. Implementación Física del Sistema

El despliegue experimental se validó mediante un prototipo funcional montado en matriz de contactos para pruebas de continuidad eléctrica, escalado posteriormente a condiciones de sustrato controlado en invernadero real.

### Prototipado Electrónico (Banco de Pruebas)
En esta fase se implementó una fuente conmutada comercial Mean Well S-100-12 ($12\text{V}$, $8.5\text{A}$) dedicada a energizar la sonda industrial NPK, mientras que se derivó una línea regulada de $5\text{V}$ para la alimentación del bus lógico del ESP32 y el transceptor MAX485. 

![Implementación en Protoboard](docs/images/hw_implementation.jpg)

### Instrumentación en Entorno Real de Cultivo
Para evitar el sesgo físico provocado por la radiación lumínica ambiental externa (ruido óptico en las lecturas de los fotodiodos), se diseñó e implementó una **cámara de aislamiento óptico sellada** (estructura cilíndrica negra) acoplada directamente sobre el sensor AS7262. Esto garantiza que la firma espectral capturada responda únicamente a la reflectancia inducida por el LED de excitación del propio sensor sobre la morfología foliar de la lechuga.

![Implementación Física en Cultivo](docs/images/physical_implementation.jpg)

---

## 📊 4. Pipeline de Modelado y Ejemplos de Pruebas

El núcleo analítico procesa las características físicas adquiridas para discriminar entre 4 estados nutricionales: `Control (Sano)`, `Déficit de N`, `Déficit de P`, y `Déficit de K`.

### Extracción de Características (Feature Engineering)
Además de las lecturas directas de concentración en $\text{mg/kg}$ de la sonda y los conteos por canal espectral ($450\text{nm}$ a $650\text{nm}$), el pipeline genera variables sintéticas basadas en razones espectrales no lineales para maximizar la separabilidad de las clases:

$$\text{Índice Razón Espectral} = \frac{\text{Banda}_{\lambda_1}}{\text{Banda}_{\lambda_2}}$$

### Ejemplo de Estructura de Datos de Entrada (Raw Telemetry)
Cada fila representa un vector de estado consolidado enviado por el firmware hacia el pipeline de ejecución:

| Sonda_N | Sonda_P | Sonda_K | Ch_450nm | Ch_500nm | Ch_550nm | Ch_570nm | Ch_600nm | Ch_650nm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 45.2 | 12.1 | 33.4 | 1204.5 | 980.2 | 2341.1 | 1890.4 | 540.3 | 310.2 |

### Log de Ejecución del Pipeline Predictivo (`train_pipeline.py`)
A continuación se detalla una salida estándar de la validación cruzada y el proceso de evaluación del pipeline predictivo implementado en `ml_pipeline/scripts/`:

```bash
$ python ml_pipeline/scripts/train_pipeline.py

[INFO] Cargando registros desde data/raw/dataset_lechugas.csv...
[INFO] Filas cargadas: 1,420 | Columnas detectadas: 9
[INFO] Ejecutando Preprocesamiento: Imputación de nulos y remoción de Outliers (Z-score > 3)
[INFO] Extracción de Características: Calculando índices de reflectancia foliar...
[INFO] Aplicando Feature Scaling mediante StandardScaler robusto.
[INFO] Entrenando Clasificador Ensemble (Gradient Boosting Classifier)...

=== EVALUACIÓN DEL MODELO (Validación Cruzada K-Fold, K=5) ===
Promedio de Exactitud (Accuracy): 94.62% (+/- 1.15%)

Reporte de Clasificación Final:
              precision    recall  f1-score   support

     Control       0.96      0.95      0.95       284
   Deficit_N       0.93      0.94      0.93       284
   Deficit_P       0.95      0.92      0.93       284
   Deficit_K       0.94      0.97      0.95       284

    accuracy                           0.95      1136
   macro avg       0.95      0.95      0.94      1136
weighted avg       0.95      0.95      0.94      1136

[SUCCESS] Pipeline ejecutado con éxito. Modelos exportados a 'ml_pipeline/models/model.joblib'