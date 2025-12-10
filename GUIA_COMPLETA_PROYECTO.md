# 🎓 GUÍA COMPLETA DEL PROYECTO MLP-MNIST CONCURRENTE

**Estudiante**: Carlos Gómez  
**Fecha**: 8 de diciembre de 2025  
**Rama actual**: `dev` (C + OpenMP + Frontend/Backend)  
**Compañero**: Rama `devS` (Python secuencial + multiprocessing)

---

## 📊 RESUMEN EJECUTIVO

### ¿Qué es este proyecto?

Es una implementación **desde cero** de una **Red Neuronal MLP** _(Multilayer Perceptron = Perceptrón Multicapa: red neuronal artificial con múltiples capas de neuronas conectadas)_ para clasificar dígitos escritos a mano del **dataset MNIST** _(Modified National Institute of Standards and Technology: colección de 70,000 imágenes de dígitos del 0-9 escritos a mano, estándar para aprender Machine Learning)_.

El objetivo **NO es la precisión** _(lograr el mayor % de aciertos)_, sino **comparar el rendimiento** _(velocidad de ejecución)_ **de diferentes paradigmas de programación concurrente** _(formas de ejecutar código en paralelo: con hilos, procesos, GPU, etc.)_.

### Arquitectura de la Red Neuronal

```
ENTRADA (784 neuronas)  →  OCULTA (512 neuronas, ReLU)  →  SALIDA (10 neuronas, Softmax)
     28x28 píxeles              Aprende patrones              0,1,2,3,4,5,6,7,8,9
```

**Explicación de componentes**:

- **784 neuronas de entrada**: Cada píxel de la imagen 28×28 = 784 valores
- **ReLU** _(Rectified Linear Unit)_: Función de activación que convierte negativos en cero: `f(x) = max(0, x)`. Ayuda a la red a aprender patrones no lineales
- **Softmax**: Función que convierte números en probabilidades que suman 100%. Ej: [0.05, 0.02, 0.87, ...] = 5% es un "0", 2% es un "1", 87% es un "2"

### Implementaciones Requeridas (6 versiones)

| #   | Versión                | Estado           | Responsable | Speedup Esperado   |
| --- | ---------------------- | ---------------- | ----------- | ------------------ |
| 1a  | Python Secuencial      | ✅ **INTEGRADO** | Compañero   | Baseline (1.0×)    |
| 1b  | **C Secuencial**       | ✅ **TÚ**        | **TÚ**      | 2-3× vs Python     |
| 2a  | Python Multiprocessing | ✅ **INTEGRADO** | Compañero   | 2-4× vs Python seq |
| 2b  | **C + OpenMP**         | ✅ **TÚ**        | **TÚ**      | **4-8× vs C seq**  |
| 3a  | CUDA (C++)             | ⏳ Pendiente     | Ambos       | 10-50×             |
| 3b  | PyCUDA (Python)        | ⏳ Pendiente     | Ambos       | 8-30×              |

**Glosario de términos**:

- **Baseline** _(línea base)_: Versión de referencia para comparar. Su speedup es 1.0× (se compara consigo misma)
- **Speedup** _(aceleración)_: Cuánto más rápido corre. Ej: 4× = 4 veces más rápido = tarda 1/4 del tiempo
- **Secuencial**: Código que ejecuta una instrucción a la vez (sin paralelismo)
- **Multiprocessing**: Paralelismo usando múltiples procesos separados (cada uno con su propia memoria)
- **OpenMP** _(Open Multi-Processing)_: Librería para paralelizar código C/C++ usando hilos (threads que comparten memoria)
- **CUDA**: Plataforma de NVIDIA para programar GPUs (miles de núcleos pequeños trabajando juntos)
- **PyCUDA**: Versión de CUDA para Python

---

## 🏗️ ARQUITECTURA TÉCNICA

### Backend (Integrado - rama `dev`)

#### 1. **Python Secuencial** (`backend/py_secuencial/`)

- **Propósito**: Baseline _(versión de referencia)_ en Python, implementación estándar sin optimizaciones
- **Componentes**:
  - `src/mlp.py`: Clase MLP con **Forward** _(calcular predicción)_ y **Backward** _(calcular errores para aprender)_
  - `src/data_loader.py`: Carga MNIST desde formatos **IDX** _(formato original del dataset)_ o **BIN** _(formato binario personalizado para C)_
  - `src/train.py`: **Loop de entrenamiento** _(ciclo que repite el proceso de aprender epoch por epoch)_
- **Ejecutar**:

```bash
cd backend/py_secuencial/src
python train.py --epochs 10 --batch-size 256
# --epochs: número de veces que la red ve TODO el dataset (10 pasadas completas)
# --batch-size: cuántas imágenes procesar juntas antes de actualizar pesos (256 imágenes a la vez)
```

#### 2. **Python Multiprocessing** (`backend/py_multiprocessing/`)

- **Propósito**: Paralelización con **procesos** _(programas separados que NO comparten memoria)_ = memoria distribuida
- **Estrategia**: División de **mini-batches** _(pequeños grupos de imágenes)_ entre **workers** _(procesos trabajadores)_
- **Ejecutar**:

```bash
cd backend/py_multiprocessing/src
python train_mp.py --epochs 10 --workers 4
# --workers: número de procesos paralelos (4 = usa 4 núcleos de CPU)
```

#### 3. **C Secuencial** (`backend/c_secuencial/`)

- **Propósito**: Baseline en C, más rápido que Python (código compilado) pero sin paralelización
- **Componentes**:
  - `include/matrix.h`: Multiplicación de matrices (**GEMM** = _General Matrix Multiply_, operación matemática más costosa de la red)\*
  - `include/mlp.h`: **Forward propagation** _(calcular predicción capa por capa)_ y **Backward propagation** _(calcular gradientes = dirección del error para corregir pesos)_
  - `include/data.h`: Carga de archivos `.bin` _(binarios con las imágenes preprocesadas)_
  - `src/train.c`: Loop de entrenamiento (10 **epochs** = _pasadas completas por el dataset_)
  - `src/export_weights.c`: Exporta **pesos** _(parámetros aprendidos W1, W2, b1, b2)_ a **JSON** _(formato legible para JavaScript)_ para el frontend

**Compilar y ejecutar**:

```bash
cd backend/c_secuencial
make
./bin/train_seq.exe
```

**Salida**:

- `backend/results/raw/c_sequential.csv` (métricas por época)
- `backend/api/model_weights_sequential.json` (pesos para frontend)

#### 4. **C + OpenMP** (`backend/c_openmp/`)

- **Propósito**: Paralelización con **hilos** _(threads: mini-procesos livianos que comparten la misma memoria)_ = memoria compartida
- **Optimizaciones**:
  - `#pragma omp parallel for`: **Directiva** _(instrucción especial)_ de OpenMP que divide un bucle entre varios hilos automáticamente
  - Paralelización del **batch processing** _(procesar múltiples lotes de imágenes simultáneamente)_
  - Uso de `OMP_NUM_THREADS`: **Variable de entorno** _(configuración del sistema)_ que controla cuántos hilos usar (ej: 8 = usar 8 núcleos de CPU)

**Compilar y ejecutar**:

```bash
cd backend/c_openmp
make
export OMP_NUM_THREADS=8  # En Windows: set OMP_NUM_THREADS=8
./bin/train_openmp.exe
```

**Salida**:

- `backend/results/raw/c_openmp.csv`
- `backend/api/model_weights_openmp.json`

#### 5. **API Node.js** (`backend/api/`)

- **Propósito**: Servidor **REST** _(Representational State Transfer: estilo de comunicación web donde el cliente hace peticiones HTTP)_ para que el frontend haga predicciones
- **Endpoints** _(URLs específicas que el servidor entiende)_:
  - `GET /api/health`: **GET** _(solicitar información)_ para verificar si el servidor está funcionando
  - `GET /api/models`: Listar modelos disponibles (sequential, openmp)
  - `POST /api/predict`: **POST** _(enviar datos)_ una imagen y recibir la predicción del dígito

**Iniciar**:

```bash
cd backend/api
npm install
npm start  # Puerto 3001
```

#### 6. **Frontend React** (`frontend/`)

- **Propósito**: Interfaz gráfica de usuario para dibujar y predecir dígitos
- **Características**:
  - **Canvas** _(lienzo HTML5)_: Área de dibujo que captura trazos del mouse y los convierte a imagen 28×28 píxeles
  - Selección de modelo (Sequential/OpenMP): Dropdown para elegir qué versión de la red usar
  - Visualización de **probabilidades** _(% de confianza de cada dígito 0-9)_: Gráfico de barras mostrando qué tan segura está la red

**Iniciar**:

```bash
cd frontend
npm install
npm run dev  # Puerto 5173
```

### Dataset (`backend/data/mnist/`)

**Archivos generados por scripts Python**:

- `train_images.bin`: 60,000 imágenes para entrenar (180 MB)
- `train_labels.bin`: 60,000 **etiquetas one-hot** _(representación donde el dígito correcto es 1 y el resto 0. Ej: "3" = [0,0,0,1,0,0,0,0,0,0])_ (2.4 MB)
- `test_images.bin`: 10,000 imágenes para validar (30 MB)
- `test_labels.bin`: 10,000 etiquetas one-hot para validación (0.4 MB)

**⚠️ Estos archivos NO están en el repositorio** (son generados localmente).

---

## 🔄 GESTIÓN DE GIT

### Estado Actual

**Tu rama `dev` (✅ ACTUALIZADA)**:

- ✅ Código Python integrado desde `devS`
- ✅ Frontend completo y funcional
- ✅ API refactorizada para múltiples modelos
- ✅ Exportación de pesos mejorada
- ✅ Archivos binarios excluidos de Git
- ✅ Todo pusheado exitosamente a GitHub

**Estructura Completa**:

- `backend/py_secuencial/` - Python baseline
- `backend/py_multiprocessing/` - Python paralelo
- `backend/c_secuencial/` - C baseline
- `backend/c_openmp/` - C paralelo (4.45× speedup)
- `backend/api/` - Node.js REST API
- `frontend/` - React UI

### Integración Completada ✅

**Lo que se hizo**:

```bash
# ✅ PASO 1: Guardado de trabajo
git add .
git commit -m "feat: Frontend React + API Node.js + exportación mejorada"

# ✅ PASO 2: Limpieza de archivos binarios
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch backend/data/mnist/*.bin backend/data/mnist/*ubyte'
git push --force-with-lease origin dev

# ✅ PASO 3: Integración de código Python (cherry-pick)
git checkout -b dev-integration
git checkout origin/devS -- py_secuencial py_multiprocessing
mv py_secuencial backend/
mv py_multiprocessing backend/
# Ajuste de rutas en archivos Python
git add backend/py_*
git commit -m "feat: Integrar Python desde devS"

# ✅ PASO 4: Merge y push
git checkout dev
git merge dev-integration
git push origin dev
```

**Resultado**: Todas las implementaciones Python están en `backend/` con rutas corregidas.

---

## 📐 MATEMÁTICAS DEL MLP

### Forward Propagation (Predicción)

**Notación**:

- `@` = multiplicación de matrices
- `(batch, 512)` = dimensiones de la matriz (filas, columnas)
- `W1, W2` = matrices de pesos (parámetros aprendidos)
- `b1, b2` = vectores de bias (desplazamiento aprendido)

```
1. Z1 = X @ W1 + b1        # (batch, 512) = (batch, 784) @ (784, 512) + (512,)
   # Cada imagen (784 píxeles) se multiplica por pesos W1 para obtener 512 valores

2. A1 = ReLU(Z1)           # A1[i] = max(0, Z1[i])
   # ReLU convierte negativos en cero, mantiene positivos

3. Z2 = A1 @ W2 + b2       # (batch, 10) = (batch, 512) @ (512, 10) + (10,)
   # 512 valores se multiplican por pesos W2 para obtener 10 valores (uno por dígito)

4. A2 = Softmax(Z2)        # A2[j] = exp(Z2[j]) / sum(exp(Z2))
   # Softmax convierte los 10 valores en probabilidades que suman 1.0 (100%)
```

### Backward Propagation (Aprendizaje)

**Notación**:

- `dZ, dW, db` = **gradientes** _(derivadas que indican cuánto cambiar cada parámetro)_
- `^T` = **transpuesta** _(voltear filas y columnas de una matriz)_
- `⊙` = multiplicación elemento a elemento (Hadamard)
- `Y_true` = etiqueta correcta (respuesta esperada)

```
1. dZ2 = A2 - Y_true          # (batch, 10) Error en la salida
   # Diferencia entre predicción (A2) y realidad (Y_true)

2. dW2 = A1^T @ dZ2 / batch   # (512, 10) Gradiente de W2
   # Calcula cuánto contribuyó cada peso W2 al error

3. db2 = sum(dZ2) / batch     # (10,) Gradiente de b2
   # Suma de errores para cada neurona de salida

4. dA1 = dZ2 @ W2^T           # (batch, 512) Error propagado hacia atrás
   # Distribuye el error de salida hacia la capa oculta

5. dZ1 = dA1 ⊙ ReLU'(Z1)     # (batch, 512) ⊙ = multiplicación elemento a elemento
   # ReLU'(x) = 1 si x>0, 0 si x≤0 (derivada de ReLU)

6. dW1 = X^T @ dZ1 / batch    # (784, 512) Gradiente de W1
   # Calcula cuánto contribuyó cada peso W1 al error

7. db1 = sum(dZ1) / batch     # (512,) Gradiente de b1
   # Suma de errores para cada neurona oculta
```

### Actualización de Pesos

**α (alpha)** = **learning rate** _(tasa de aprendizaje)_: qué tan grande es cada paso de corrección

- Si α es muy grande (ej: 1.0) → aprende rápido pero puede pasarse
- Si α es muy pequeño (ej: 0.0001) → aprende lento pero con precisión
- 0.01 es un buen balance

```
W1 = W1 - α * dW1   # Resta el gradiente escalado por α
                     # Ejemplo: si dW1=2 y α=0.01, resta 0.02
b1 = b1 - α * db1   # Lo mismo para bias
W2 = W2 - α * dW2
b2 = b2 - α * db2
```

**Intuición**: Los gradientes indican "hacia dónde subir el error", así que restamos para bajar el error.

### Cuello de Botella Computacional

**El 95% del tiempo está en la multiplicación de matrices**:

- `X @ W1`: (batch, 784) × (784, 512) = **401,408 operaciones/imagen**
- `A1 @ W2`: (batch, 512) × (512, 10) = **5,120 operaciones/imagen**

**Por eso se paraleliza la multiplicación de matrices**.

---

## ⚡ PARALELIZACIÓN (Tu contribución principal)

### OpenMP: Estrategias Implementadas

#### 1. Paralelización de GEMM (General Matrix Multiply)

```c
// ANTES (Secuencial)
for (int i = 0; i < M; i++) {              // M filas
    for (int j = 0; j < N; j++) {          // N columnas
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {      // K elementos a sumar
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;                   // Resultado en C[i][j]
    }
}
// Esto ejecuta M×N×K operaciones en serie (uno tras otro)

// DESPUÉS (Paralelo con OpenMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
// OpenMP divide el trabajo entre múltiples hilos automáticamente
```

**Explicación de directivas OpenMP**:

- **`#pragma omp parallel for`**: Directiva que dice "divide este bucle entre varios hilos"
- **`collapse(2)`**: Combina los 2 loops externos (i y j) en uno solo → más iteraciones = mejor distribución entre hilos
  - Sin collapse: 512 iteraciones (solo i)
  - Con collapse(2): 512×10 = 5,120 iteraciones (i×j)
- **`schedule(dynamic)`**: Estrategia de distribución dinámica
  - **static** _(estático)_: Divide las iteraciones equitativamente al inicio (rápido pero puede desbalancearse)
  - **dynamic** _(dinámico)_: Los hilos toman trabajo según terminan (mejor balance, pequeño overhead)

#### 2. Reducción Paralela para Gradientes

```c
#pragma omp parallel for reduction(+:db2[:OUTPUT_SIZE])
for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        db2[j] += gradients[i * OUTPUT_SIZE + j];
    }
}
// Suma los gradientes de todas las imágenes del batch
```

**Explicación de `reduction`**:

- **Problema sin reduction**: Si múltiples hilos suman a `db2[j]` simultáneamente → **race condition** _(conflicto: dos hilos leen/escriben al mismo tiempo, resultado incorrecto)_
- **Solución con reduction**:
  1. Cada hilo crea su propia copia privada de `db2`
  2. Cada hilo suma en su copia (sin conflictos)
  3. Al final, OpenMP combina todas las copias sumándolas
- **`reduction(+:db2[:OUTPUT_SIZE])`**: Operador `+` (suma), variable `db2`, tamaño `OUTPUT_SIZE` (10 elementos)

**Alternativa sin reduction** _(más lenta)_:

````c
#pragma omp parallel for
for (int i = 0; i < batch_size; i++) {
    #pragma omp critical  // Solo un hilo a la vez puede entrar aquí
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        db2[j] += gradients[i * OUTPUT_SIZE + j];
    }
}
// Critical crea un cuello de botella (serializa el trabajo)

### Escalabilidad Medida

| Hilos | Tiempo (s) | Speedup   | Eficiencia |
| ----- | ---------- | --------- | ---------- |
| 1     | 1539       | 1.0×      | 100%       |
| 2     | 820        | 1.88×     | 94%        |
| 4     | 450        | 3.42×     | 86%        |
| 8     | 346        | **4.45×** | 56%        |

**Cómo se calculan**:
- **Speedup** = Tiempo(1 hilo) / Tiempo(N hilos)
  - Ej: 1539s / 346s = 4.45×
- **Eficiencia** = Speedup / Número de hilos × 100%
  - Ej: 4.45 / 8 × 100% = 56%
  - **Eficiencia 100%** = speedup lineal ideal (doblar hilos = mitad de tiempo)
  - **Eficiencia <100%** = hay partes que no se pueden paralelizar + overhead

**Observación**: La eficiencia baja al aumentar hilos debido a:
1. **Ley de Amdahl**: Siempre hay una porción secuencial (S) que no se paraleliza
   - Speedup máximo = 1 / S
   - Si 5% es secuencial → speedup máximo = 1/0.05 = 20×
2. **Overhead** *(costo extra)*: Crear hilos, sincronizar, combinar resultados
3. **Contención de memoria** *(cuellos de botella)*: Múltiples hilos accediendo a la misma RAM

---

## 🧪 EXPERIMENTOS Y MÉTRICAS

### Métricas a Recopilar

Para cada implementación, registra en `results/raw/<nombre>.csv`:

```csv
epoch,train_loss,train_accuracy,test_accuracy,time_seconds
1,0.532,0.842,0.838,154.3
2,0.321,0.906,0.901,152.1
...
````

### Gráficas del Informe

#### 1. Speedup vs. Número de Hilos (OpenMP)

```
Speedup
   8×  ┤
   7×  ┤                           ╱ Ideal (lineal)
   6×  ┤                        ╱
   5×  ┤                     ╱
   4×  ┤                  ╱•───── Real (4.45×)
   3×  ┤               ╱•
   2×  ┤            ╱•
   1×  ┤•────────•
       └─────────────────────────
       1  2  4  6  8  10  12  14  Hilos
```

**Análisis**: ¿Por qué no es lineal? (Overhead, sincronización, Amdahl)

#### 2. Comparación de Tiempos

```
Tiempo (segundos)
2000 ┤ ████████████████████  Python Seq (1800s)
1500 ┤ █████████████████     C Seq (1539s)
1000 ┤ █████████             Python MP (900s)
 500 ┤ ████                  C OpenMP (346s)
   0 ┤
```

#### 3. Profiling GPU (futuro)

```
Transfer CPU→GPU:  15% (120ms)
Kernel Execution:  70% (560ms)
Transfer GPU→CPU:  10% (80ms)
Overhead:           5% (40ms)
```

---

## 🎤 GUÍA DE SUSTENTACIÓN

### ¿Necesitas Frontend/Backend?

**Para la sustentación académica: NO ES OBLIGATORIO**

El proyecto requiere:

1. ✅ **Código fuente** de las 6 implementaciones
2. ✅ **Informe técnico** con métricas y análisis
3. ✅ **Presentación oral** (10-15 min)

**El frontend es un EXTRA** que demuestra que el modelo funciona, pero puedes sustentar solo con:

- Terminal mostrando el entrenamiento
- CSVs con métricas
- Gráficas en el informe

### Cómo Usar Todo Manualmente (sin Frontend)

#### Escenario 1: Solo entrenar y ver métricas

```bash
# 1. Entrenar modelo C Secuencial
cd backend/c_secuencial
make
./bin/train_seq.exe

# Verás en consola:
# Epoch 1/10: Loss=0.532, Train Acc=84.2%, Test Acc=83.8% (154s)
# Epoch 2/10: Loss=0.321, Train Acc=90.6%, Test Acc=90.1% (152s)
# ...

# 2. Entrenar modelo C OpenMP (8 hilos)
cd ../c_openmp
make
set OMP_NUM_THREADS=8
./bin/train_openmp.exe

# 3. Ver resultados en CSV
cat backend/results/raw/c_sequential.csv
cat backend/results/raw/c_openmp.csv

# 4. Generar gráficas con Python
cd backend/scripts
python plot_results.py  # (si tienes este script)
```

#### Escenario 2: Mostrar predicciones en vivo

**Opción A: Con Frontend (más bonito)**

```bash
# Terminal 1: API
cd backend/api
npm start

# Terminal 2: Frontend
cd frontend
npm run dev

# Navegador: http://localhost:5173
# Dibujas un "7" → Modelo predice "7 (95%)"
```

**Opción B: Sin Frontend (solo API + curl)**

```bash
# Terminal 1: API
cd backend/api
npm start

# Terminal 2: Probar predicción
curl -X POST http://localhost:3001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "modelId": "openmp",
    "image": [0,0,0,...,255,...,0]  # Array de 784 valores (0-255)
  }'

# Respuesta:
# {"success":true,"prediction":7,"probabilities":[0.01,0.02,...,0.95,...]}
```

**Opción C: Solo línea de comandos (test.c)**

Crea un pequeño programa que cargue los pesos y prediga una imagen:

```c
// test_prediction.c
#include "mlp.h"
#include "data.h"

int main() {
    MLP mlp;
    mlp_load_weights(&mlp, "model_weights_openmp.json");

    // Cargar imagen de prueba (por ej. la #0 del test set)
    float image[784];
    load_test_image(0, image);

    // Predecir
    float output[10];
    mlp_forward(&mlp, image, output);

    // Mostrar
    int predicted = argmax(output, 10);
    printf("Imagen #0: Predicción = %d (confianza: %.2f%%)\n",
           predicted, output[predicted]*100);
}
```

### Estructura de la Presentación (10-15 min)

#### Diapositiva 1: Portada

- Título del proyecto
- Nombres
- Fecha

#### Diapositiva 2: Contexto

- ¿Por qué Deep Learning es costoso?
- Necesidad de paralelización
- Objetivo: Comparar paradigmas

#### Diapositiva 3: Arquitectura MLP

- Diagrama de 3 capas
- Ecuaciones Forward/Backward
- Cuello de botella: GEMM

#### Diapositiva 4: Implementaciones

- Tabla con las 6 versiones
- Estado actual (4/6 completadas)

#### Diapositiva 5: Metodología

- Hardware usado (CPU, RAM, núcleos)
- Hiperparámetros fijos
- Cómo se midió el tiempo

#### Diapositiva 6: Resultados - Tabla Comparativa

```
| Versión           | Tiempo | Speedup | Accuracy |
|-------------------|--------|---------|----------|
| Python Seq        | 1800s  | 1.0×    | 93.2%    |
| C Seq             | 1539s  | 1.17×   | 93.5%    |
| Python MP (4 proc)| 900s   | 2.0×    | 93.2%    |
| C OpenMP (8 hilos)| 346s   | 5.2×    | 93.5%    |
```

#### Diapositiva 7: Gráfica Speedup OpenMP

- Curva Real vs. Ideal
- Análisis de Amdahl

#### Diapositiva 8: Demo en Vivo (opcional)

- Mostrar frontend prediciendo un dígito
- O ejecutar en terminal

#### Diapositiva 9: Conclusilei esto, ahoravamos probar que todo funioones

- OpenMP logró 4.45× con 8 hilos
- Multiprocessing tiene overhead de IPC
- GPU (futuro) promete 10-50×

#### Diapositiva 10: Preguntas

---

## 🛠️ CÓMO CORRER TODO PASO A PASO

### Pre-requisitos

**Windows (MSYS2)**:

```bash
# Ya debes tener instalado (según INSTALL_C_TOOLS.md):
- GCC con OpenMP
- Make
- Node.js + npm
- Python 3.8+
```

### Paso 1: Descargar y preprocesar MNIST

```bash
cd backend/scripts
python download_mnist.py
python preprocess_for_c.py

# Verifica:
ls backend/data/mnist/
# Deberías ver: train_images.bin, train_labels.bin, etc.
```

### Paso 2: Entrenar modelos C

```bash
# Secuencial
cd backend/c_secuencial
make clean
make
./bin/train_seq.exe  # Toma ~25 min

# OpenMP (8 hilos)
cd ../c_openmp
make clean
make
set OMP_NUM_THREADS=8
./bin/train_openmp.exe  # Toma ~6 min
```

### Paso 3: Verificar exportación de pesos

```bash
# Deben existir:
ls backend/api/model_weights_sequential.json
ls backend/api/model_weights_openmp.json
```

### Paso 4: Levantar el stack completo

```bash
# Terminal 1: API
cd backend/api
npm install
npm start  # Puerto 3001

# Terminal 2: Frontend
cd frontend
npm install
npm run dev  # Puerto 5173

# Navegador:
# http://localhost:5173
```

### Paso 5: Probar predicción

1. Dibuja un dígito (ej. "5")
2. Selecciona modelo ("C OpenMP")
3. Click "Predecir"
4. Verás: "Predicción: 5 (Confianza: 92%)"

---

## 🔬 ANÁLISIS PROFUNDO

### ¿Por qué C es más rápido que Python?

1. **Compilado vs. Interpretado**: C se compila a código máquina nativo
2. **Sin GIL**: Python tiene el Global Interpreter Lock
3. **Control de memoria**: C gestiona memoria manualmente (malloc/free)
4. **Optimizaciones del compilador**: `-O3` aplica vectorización, loop unrolling

### ¿Por qué OpenMP escala bien?

1. **Memoria compartida**: Los hilos comparten W1, W2 (no hay copia)
2. **Granularidad gruesa**: Cada hilo procesa múltiples filas de la matriz
3. **Buen locality**: Accesos a memoria son secuenciales (cache-friendly)

### ¿Qué limita el Speedup? (Ley de Amdahl)

```
Speedup = 1 / (S + P/N)

S = Fracción secuencial (ej. 0.05 = 5%)
P = Fracción paralelizable (ej. 0.95 = 95%)
N = Número de hilos

Ejemplo con 8 hilos:
Speedup = 1 / (0.05 + 0.95/8) = 1 / 0.169 = 5.92×

Real: 4.45× (porque overhead de sincronización)
```

**Partes secuenciales**:

- Carga de datos
- Escritura de logs
- Actualización de pesos (tiene sección crítica)

---

## 📝 CHECKLIST DE ENTREGA

### Código Fuente ✅

```
✅ backend/py_secuencial/       (Python + NumPy)
✅ backend/py_multiprocessing/  (Python + multiprocessing)
✅ backend/c_secuencial/        (compilable con make)
✅ backend/c_openmp/            (compilable con make)
⏳ backend/pycuda_gpu/          (pendiente)
✅ frontend/                    (extra, no requerido)
✅ backend/api/                 (extra, no requerido)
```

### Informe Técnico (Word/PDF)

```
[ ] 1. Introducción (contexto, objetivos)
[ ] 2. Arquitectura MLP (diagrama, ecuaciones)
[ ] 3. Metodología (hardware, hiperparámetros)
[ ] 4. Resultados:
    [ ] 4.1 Tabla comparativa de tiempos
    [ ] 4.2 Gráfica Speedup OpenMP
    [ ] 4.3 Análisis Ley de Amdahl
    [ ] 4.4 Comparación Python MP vs. C OpenMP
    [ ] 4.5 Profiling GPU (si completan CUDA)
[ ] 5. Conclusiones
[ ] 6. Referencias
```

### Presentación (PPT/PDF)

```
[ ] 10-12 diapositivas
[ ] Máximo 15 minutos
[ ] Todos los miembros participan
[ ] Incluir gráficas del informe
```

---

## 🚨 PROBLEMAS COMUNES

### 1. "No se encuentran archivos .bin"

**Solución**:

```bash
cd backend/scripts
python preprocess_for_c.py
```

### 2. "OpenMP no compila"

**Solución**:

```bash
# Verifica que gcc soporte OpenMP:
gcc -fopenmp --version

# Si no, reinstala gcc con MSYS2:
pacman -S mingw-w64-x86_64-gcc
```

### 3. "Frontend no se conecta a la API"

**Solución**:

```bash
# Verifica que la API esté corriendo:
curl http://localhost:3001/api/health

# Si no responde, revisa que:
# 1. npm start esté corriendo en backend/api
# 2. No haya otro proceso en puerto 3001
```

### 4. "Accuracy es muy bajo (<80%)"

**Causas**:

- Pesos inicializados incorrectamente
- Learning rate muy alto/bajo
- Bug en backpropagation

**Debug**:

```bash
# Compara pesos de época 1 con implementación conocida
# Verifica que la loss disminuya cada época
```

---

## 🎯 PRÓXIMOS PASOS

### Inmediato (antes de sustentar)

1. ✅ Commitear y pushear rama `dev`
2. ✅ Integrar código Python de `devS`
3. ⏳ Completar informe técnico
4. ⏳ Crear presentación
5. ⏳ Ensayar sustentación

### Opcional (si hay tiempo)

6. ⏳ Implementar CUDA/PyCUDA
7. ⏳ Mejorar visualizaciones del frontend
8. ⏳ Agregar más gráficas al informe

---

## 📚 RECURSOS

### Documentación Interna

- `backend/docs/FORMULAS_IMPLEMENTACION.md`: Matemáticas detalladas
- `backend/docs/WORKFLOW.md`: Dependencias entre componentes
- `backend/docs/INSTALL_C_TOOLS.md`: Setup de herramientas

### Referencias Externas

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- OpenMP Tutorial: https://computing.llnl.gov/tutorials/openMP/
- Backpropagation: http://neuralnetworksanddeeplearning.com/

---

## 💡 TIPS PARA LA SUSTENTACIÓN

### 1. Demuestra que entiendes TODO

**Pregunta típica**: "¿Por qué ReLU en lugar de sigmoid?"

**Respuesta**:

> "ReLU evita el problema de vanishing gradient porque su derivada es siempre 1 (si x>0) o 0 (si x≤0). Sigmoid satura en los extremos, haciendo que la derivada sea casi cero y el aprendizaje se detenga."

### 2. Sé honesto sobre limitaciones

**Pregunta**: "¿Por qué solo lograron 4.45× con 8 hilos?"

**Respuesta**:

> "Según la Ley de Amdahl, el speedup teórico máximo es 1/(S + P/N). Estimamos que el 5% del código es secuencial (carga de datos, logs). Además, hay overhead de sincronización en las secciones críticas (actualización de pesos). Por eso no logramos el ideal de 8×."

### 3. Relaciona con el mundo real

**Pregunta**: "¿Cómo se relaciona esto con frameworks como TensorFlow?"

**Respuesta**:

> "TensorFlow usa GEMM (multiplicación de matrices) implementado en cuBLAS (GPU) o MKL (CPU). Nosotros implementamos GEMM desde cero para entender los cuellos de botella. En producción, siempre usaríamos librerías optimizadas."

### 4. Demuestra el frontend (si lo tienes)

- Es visual y impresionante
- Muestra que el modelo REALMENTE funciona
- Diferencia tu proyecto del de otros grupos

### 5. Ten métricas a la mano

- Speedup exacto (4.45×)
- Tiempo por época (346s vs. 1539s)
- Accuracy final (93.5%)

---

## ✅ CONCLUSIÓN

**Lo que YA TIENES (rama `dev`) - ✅ INTEGRADO**:

- ✅ Python Secuencial (`backend/py_secuencial/`)
- ✅ Python Multiprocessing (`backend/py_multiprocessing/`)
- ✅ C Secuencial (`backend/c_secuencial/`)
- ✅ C OpenMP (`backend/c_openmp/`) - **4.45× speedup**
- ✅ Frontend React (extra, no obligatorio)
- ✅ API Node.js (extra, no obligatorio)
- ✅ Sistema de exportación de pesos

**Lo que QUEDA POR HACER (ambos)**:

- CUDA/PyCUDA
- Informe técnico
- Presentación

**Para sustentar SOLO TU PARTE**:

1. Demuestra entrenamiento en C (secuencial y OpenMP)
2. Muestra gráfica de Speedup
3. Explica optimizaciones con OpenMP
4. (Opcional) Demo del frontend

**Tiempo estimado para preparar sustentación**: 2-3 días

---

**¿Preguntas? Revisa esta guía o consulta los archivos en `backend/docs/`**
