# Diseño Experimental - MLP MNIST Concurrente

Este documento describe el diseño experimental completo para el proyecto de implementación y paralelización de una Red Neuronal MLP desde cero para clasificación de dígitos MNIST.

## 🎯 Objetivo Principal

**NO** construir la red neuronal más precisa, sino **analizar y comparar el rendimiento** de diferentes implementaciones secuenciales y paralelas, entendiendo:

- Cuellos de botella computacionales (multiplicación de matrices)
- Técnicas de paralelismo en CPU (memoria compartida y distribuida)
- Paralelismo masivo en GPU
- Métricas: Speedup, Overhead, Ley de Amdahl
- Problemática de transferencia Host-Device en GPGPU

---

## 📊 Dataset

### MNIST

- **Entrenamiento**: 60,000 imágenes
- **Prueba**: 10,000 imágenes
- **Formato**: 28×28 píxeles en escala de grises
- **Clases**: 10 dígitos (0-9)

### Preprocesamiento (Obligatorio)

```
1. Normalización: [0, 255] → [0, 1]
2. Aplanamiento: 28×28 → vector de 784 características
3. One-hot encoding: etiquetas → vectores de tamaño 10
4. Mini-batches: dividir dataset en lotes
```

### Formato de Archivos

- **Python**: Usar NumPy arrays (`.npy` o carga directa)
- **C/C++**: Archivos binarios `.bin` (float32, contiguo en memoria)
  - Generados por Python para compatibilidad
  - Formato: `[n_samples, 784]` para X, `[n_samples, 10]` para Y

---

## 🧠 Arquitectura del MLP (FIJA)

### Estructura de Capas

```
Capa de Entrada:  784 neuronas (fijo)
      ↓
Capa Oculta:      512 neuronas (acordado)
      ↓
Capa de Salida:   10 neuronas (fijo)
```

### Funciones de Activación

- **Capa Oculta**: ReLU (Rectified Linear Unit)
  - `ReLU(x) = max(0, x)`
  - `ReLU'(x) = 1 if x > 0 else 0`
- **Capa de Salida**: Softmax
  - `Softmax(z_i) = exp(z_i) / Σ exp(z_j)`

### Función de Pérdida

- **Cross-Entropy Loss**
  - `L = -Σ y_i * log(ŷ_i)`

### Inicialización de Pesos

- **Método**: Xavier/Glorot Uniform
- **SEED fija**: `42` (para reproducibilidad)
- **Precisión**: `float32` (todas las implementaciones)

---

## ⚙️ Hiperparámetros (ACORDADOS)

### Parámetros Globales

```python
EPOCHS = 10
LEARNING_RATE = 0.01
HIDDEN_NEURONS = 512
RANDOM_SEED = 42
```

### Batch Size por Implementación

| Implementación         | Batch Size | Razón                         |
| ---------------------- | ---------- | ----------------------------- |
| Python Secuencial      | 64         | Balance memoria/velocidad CPU |
| Python Multiprocessing | 64         | Mismo que secuencial          |
| C Secuencial           | 64         | Consistencia con Python       |
| C OpenMP               | 64         | Mismo que secuencial          |
| PyCUDA (pequeño)       | 16         | Evaluar latencia GPU          |
| PyCUDA (grande)        | 512        | Evaluar throughput GPU        |

---

## 🔬 Fases de Implementación

### Fase 0: Preparación (Juntos)

**Responsables**: Ambos  
**Objetivo**: Acordar especificaciones y configurar entorno

**Tareas**:

- [x] Crear estructura del repositorio
- [ ] Definir y documentar arquitectura MLP
- [ ] Fijar hiperparámetros
- [ ] Crear scripts de descarga de MNIST
- [ ] Generar archivos binarios para C
- [ ] Definir formato de resultados CSV
- [ ] Configurar `.gitignore` (evitar subir datasets)

**Entregable**: `docs/experiment_design.md` completo

---

### Fase 1: Baseline Secuencial

#### 1A. Python Secuencial (Compañero)

**Carpeta**: `python_secuencial/`

**Módulos a implementar**:

1. **`data.py`**

   - Cargar MNIST
   - Normalizar a [0,1]
   - Aplanar imágenes (784)
   - One-hot encoding (10 clases)
   - Generador de mini-batches

2. **`model.py`**

   - Inicialización de pesos: W1(784×512), b1(512), W2(512×10), b2(10)
   - `forward(X_batch)`: calcula z1, a1, z2, a2
   - `backward(X_batch, Y_batch)`: calcula gradientes
   - `update_params(lr)`: actualiza pesos

3. **`loss.py`**

   - Cross-Entropy
   - Accuracy

4. **`train.py`**
   - Bucle de entrenamiento (10 epochs)
   - Medición con `time.perf_counter()`
   - Guardar CSV: `results/raw/python_sequential.csv`

**Validación**:

- Loss disminuye de ~2.3 a <0.5
- Accuracy final > 90%

---

#### 1B. C Secuencial (Tú)

**Carpeta**: `c_secuencial/`

**Módulos a implementar**:

1. **`matrix.c / matrix.h`**

   - Multiplicación de matrices (GEMM)
   - ReLU y ReLU' (derivada)
   - Softmax

2. **`mlp.c / mlp.h`**

   - Estructuras para W1, b1, W2, b2
   - `forward()`
   - `backward()`
   - `update_params()`

3. **`data.c / data.h`**

   - Lectura de archivos `.bin` (generados por Python)
   - Estructuras para dataset

4. **`train.c`**
   - Bucle de entrenamiento
   - Medición con `clock_gettime(CLOCK_MONOTONIC, ...)`
   - Guardar CSV: `results/raw/c_sequential.csv`

**Compilación**:

```bash
gcc -O3 -o train_seq *.c -lm
```

**Validación**:

- Loss converge similar a Python (diferencia < 1e-4)
- Sin NaN ni overflow

---

### Fase 2: Paralelismo en CPU

#### 2A. Python Multiprocessing (Compañero)

**Carpeta**: `python_multiprocessing/`

**Diseño Master-Worker**:

- **Master**: mantiene pesos, divide batches, promedia gradientes
- **Workers**: calculan gradientes en sub-lotes

**Implementación**:

- Usar `multiprocessing.Pool` o `Process + Queue`
- Función `compute_gradients(params, X_sub, Y_sub)`

**Experimentos**:

```
Procesos: 1, 2, 4, 8
Medir: tiempo total (10 epochs)
CSV: results/raw/python_multiprocessing.csv
Columnas: processes, total_time, speedup_vs_seq
```

---

#### 2B. C + OpenMP (Tú)

**Carpeta**: `c_openmp/`

**Paralelización**:

```c
// En matrix.c - GEMM
#pragma omp parallel for schedule(static)
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        float sum = 0.0f;
        for (int k = 0; k < inner; k++) {
            sum += A[i*inner + k] * B[k*cols + j];
        }
        C[i*cols + j] = sum;
    }
}
```

**Compilación**:

```bash
gcc -O3 -fopenmp -o train_omp *.c -lm
export OMP_NUM_THREADS=4
```

**Experimentos**:

```
Threads: 1, 2, 4, 8
Medir: tiempo total (10 epochs)
CSV: results/raw/c_openmp.csv
Columnas: threads, total_time, speedup_vs_c_seq
```

**Validación**:

- Speedup > 1 pero < # threads (por Ley de Amdahl)
- Resultados reproducibles (no race conditions)

---

### Fase 3: GPU con PyCUDA (Compañero)

**Carpeta**: `pycuda_gpu/`  
**Entorno**: Google Colab (GPU T4/P100)

**Módulos**:

1. **`gpu_gemm.py`**

   - Kernel CUDA para multiplicación de matrices
   - Compilar con `pycuda.compiler.SourceModule`
   - Función `gpu_gemm(A, B)`:
     - Copiar Host→Device
     - Lanzar kernel
     - Copiar Device→Host

2. **`gpu_mlp.py`**

   - MLP usando `gpu_gemm` en forward/backward

3. **`train_gpu.py`**
   - Bucle de entrenamiento
   - Medición con eventos CUDA:
     ```python
     start = cuda.Event()
     end = cuda.Event()
     start.record()
     # ... operación ...
     end.record()
     end.synchronize()
     time_ms = start.time_till(end)
     ```

**Experimentos**:

```
Batch sizes: 16, 512
Medir:
  - Tiempo total
  - Tiempo Host→Device
  - Tiempo kernel
  - Tiempo Device→Host
CSV: results/raw/pycuda_results.csv
```

**Validación**:

- Speedup GPU vs CPU > 5×
- Batch 512 más eficiente que 16

---

## 📈 Métricas y Formato de Resultados

### Estructura de CSV (Todos)

```csv
implementation,language,parallelization,workers_threads,batch_size,epochs,learning_rate,hidden_neurons,total_time_sec,avg_epoch_time,final_loss,final_accuracy,speedup_vs_baseline,notes
python_seq,python,none,1,64,10,0.01,512,45.2,4.52,0.234,0.921,1.00,baseline
python_mp,python,multiprocessing,4,64,10,0.01,512,15.3,1.53,0.235,0.920,2.95,
c_seq,c,none,1,64,10,0.01,512,8.7,0.87,0.233,0.921,1.00,c_baseline
c_openmp,c,openmp,8,64,10,0.01,512,1.9,0.19,0.234,0.920,4.58,
pycuda,python,gpu,1,512,10,0.01,512,2.1,0.21,0.235,0.919,21.52,batch_512
```

### Medición de Tiempos

**Python**:

```python
import time
start = time.perf_counter()
# ... entrenamiento ...
end = time.perf_counter()
total_time = end - start
```

**C**:

```c
#include <time.h>
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// ... entrenamiento ...
clock_gettime(CLOCK_MONOTONIC, &end);
double time_spent = (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1e9;
```

**PyCUDA**:

```python
import pycuda.driver as cuda
start_event = cuda.Event()
end_event = cuda.Event()
start_event.record()
# ... operación ...
end_event.record()
end_event.synchronize()
time_ms = start_event.time_till(end_event)
```

---

## 📊 Análisis y Visualización

### Scripts (Carpeta `scripts/`)

1. **`download_mnist.py`**

   - Descargar dataset automáticamente
   - Guardar en `data/mnist/`

2. **`preprocess_data.py`**

   - Generar archivos `.bin` para C
   - Normalizar y formatear datos

3. **`validate_implementation.py`**

   - Comparar outputs Python vs C (con mismo seed)
   - Verificar diferencias < 1e-4

4. **`aggregate_results.py`**

   - Leer todos los CSV de `results/raw/`
   - Generar tabla consolidada

5. **`plot_results.py`**

   - Gráfica: Speedup C+OpenMP vs #threads
   - Gráfica: Speedup Python multiprocessing
   - Gráfica: Tiempos PyCUDA desglosados (H→D, kernel, D→H)
   - Gráfica: Batch size 16 vs 512 en GPU
   - Gráfica: Comparación global de todas las implementaciones
   - Guardar en `results/figures/`

6. **`run_all_experiments.sh`**
   - Automatizar todas las ejecuciones

### Gráficas Requeridas

1. **Speedup vs Threads (OpenMP)**
2. **Speedup vs Procesos (Multiprocessing)**
3. **Comparación de Tiempos Absolutos (todas las implementaciones)**
4. **Desglose GPU (H→D, Kernel, D→H)**
5. **Batch Size Impact en GPU**
6. **Análisis de Ley de Amdahl**

---

## ✅ Validación y Reproducibilidad

### Criterios de Éxito por Fase

**Fase 1 - Baseline**:

- [ ] Loss converge de ~2.3 a <0.5 en 10 epochs
- [ ] Accuracy final > 90%
- [ ] Python y C producen resultados similares (diff < 1e-4)
- [ ] Tiempos registrados correctamente en CSV

**Fase 2 - CPU Paralelo**:

- [ ] Speedup > 1 para todas las configuraciones
- [ ] Speedup < # workers/threads (overhead + Amdahl)
- [ ] Resultados reproducibles (sin race conditions)
- [ ] Ley de Amdahl observable en gráficas

**Fase 3 - GPU**:

- [ ] Speedup GPU vs CPU baseline > 5×
- [ ] Batch 512 más rápido que batch 16
- [ ] Tiempo kernel > tiempo transferencia
- [ ] Sin errores de memoria GPU

### Protocolo de Validación Cruzada

1. Ejecutar Python secuencial con seed=42
2. Guardar pesos finales en `results/raw/weights/python_seq_weights.npy`
3. Ejecutar C secuencial con mismo seed
4. Comparar pesos finales:
   ```python
   diff = np.abs(weights_python - weights_c)
   assert np.max(diff) < 1e-4
   ```

---

## 🚫 Restricciones Fundamentales

### NO Permitido

- ❌ TensorFlow, Keras, PyTorch, Caffe
- ❌ Librerías de Deep Learning pre-construidas
- ❌ BLAS/LAPACK en C (implementar GEMM manual)

### Permitido

- ✅ **Python**: NumPy, multiprocessing, PyCUDA, matplotlib
- ✅ **C/C++**: OpenMP, CUDA, librerías estándar (stdio, stdlib, math)

---

## 📅 Cronograma Sugerido

### Semana 1

- Fase 0 completa (setup + documentación)
- Python secuencial implementado y validado
- C secuencial iniciado

### Semana 2

- C secuencial completo
- Validación cruzada Python vs C
- Inicio de paralelismo (OpenMP + multiprocessing)

### Semana 3

- OpenMP completo (experimentos con múltiples threads)
- Multiprocessing completo (experimentos con múltiples procesos)
- Scripts de análisis (`aggregate_results.py`, `plot_results.py`)

### Semana 4

- PyCUDA implementado y ejecutado en Colab
- Experimentos con batch 16 y 512
- Generación de todas las gráficas

### Semana 5 (Buffer)

- Debugging y ajustes finales
- Redacción del informe técnico
- Preparación de presentación oral

---

## 📦 Estructura de Archivos Final

```
mlp-mnist-concurrente/
├── README.md
├── docs/
│   └── experiment_design.md          # Este documento
├── data/
│   └── mnist/
│       ├── train-images.bin
│       ├── train-labels.bin
│       ├── test-images.bin
│       └── test-labels.bin
├── python_secuencial/
│   ├── data.py
│   ├── model.py
│   ├── loss.py
│   └── train.py
├── python_multiprocessing/
│   ├── data.py
│   ├── model_parallel.py
│   └── train_parallel.py
├── pycuda_gpu/
│   ├── gpu_gemm.py
│   ├── gpu_mlp.py
│   └── train_gpu.py
├── c_secuencial/
│   ├── include/
│   │   ├── matrix.h
│   │   ├── mlp.h
│   │   └── data.h
│   ├── src/
│   │   ├── matrix.c
│   │   ├── mlp.c
│   │   ├── data.c
│   │   └── train.c
│   └── Makefile
├── c_openmp/
│   ├── include/
│   │   ├── matrix.h
│   │   ├── mlp.h
│   │   └── data.h
│   ├── src/
│   │   ├── matrix_omp.c
│   │   ├── mlp_omp.c
│   │   ├── data.c
│   │   └── train_omp.c
│   └── Makefile
├── scripts/
│   ├── download_mnist.py
│   ├── preprocess_data.py
│   ├── validate_implementation.py
│   ├── aggregate_results.py
│   ├── plot_results.py
│   └── run_all_experiments.sh
└── results/
    ├── raw/
    │   ├── python_sequential.csv
    │   ├── python_multiprocessing.csv
    │   ├── c_sequential.csv
    │   ├── c_openmp.csv
    │   ├── pycuda_results.csv
    │   └── weights/
    │       ├── python_seq_final.npy
    │       └── c_seq_final.bin
    └── figures/
        ├── speedup_openmp.png
        ├── speedup_multiprocessing.png
        ├── comparison_all.png
        ├── gpu_breakdown.png
        ├── batch_size_comparison.png
        └── amdahl_analysis.png
```

---

## 👥 División de Responsabilidades

### Compañero (Python + PyCUDA)

- ✅ Python secuencial completo
- ✅ Python multiprocessing (master-worker)
- ✅ PyCUDA (kernels CUDA + experimentos en Colab)
- ✅ Scripts de descarga y preprocesamiento de datos
- ✅ Generación de archivos `.bin` para C
- ✅ Secciones del informe: metodología Python, resultados GPU
- ✅ Presentación: MLP básico, multiprocessing, PyCUDA

### Tú (C + OpenMP)

- ✅ C secuencial completo (GEMM manual, MLP from scratch)
- ✅ C OpenMP (paralelización de bucles críticos)
- ✅ Validación de convergencia numérica
- ✅ Experimentos con múltiples threads
- ✅ Secciones del informe: C implementation, OpenMP, análisis Amdahl
- ✅ Presentación: C secuencial, OpenMP speedup, comparación CPU vs GPU

### Ambos (Colaborativo)

- ✅ Fase 0: definición de arquitectura e hiperparámetros
- ✅ Scripts de análisis y visualización
- ✅ Validación cruzada de implementaciones
- ✅ Generación de gráficas finales
- ✅ Revisión del informe completo
- ✅ Ensayo de presentación

---

## 📚 Fundamentación Matemática (Referencia)

### Forward Propagation

```
Z1 = X @ W1 + b1          # (batch, 784) @ (784, 512) = (batch, 512)
A1 = ReLU(Z1)             # (batch, 512)
Z2 = A1 @ W2 + b2         # (batch, 512) @ (512, 10) = (batch, 10)
A2 = Softmax(Z2)          # (batch, 10) - probabilidades
```

### Loss

```
L = -Σ (Y * log(A2)) / batch_size
```

### Backward Propagation

```
dZ2 = A2 - Y              # (batch, 10)
dW2 = A1^T @ dZ2          # (512, batch) @ (batch, 10) = (512, 10)
db2 = sum(dZ2, axis=0)    # (10,)

dA1 = dZ2 @ W2^T          # (batch, 10) @ (10, 512) = (batch, 512)
dZ1 = dA1 * ReLU'(Z1)     # (batch, 512) element-wise
dW1 = X^T @ dZ1           # (784, batch) @ (batch, 512) = (784, 512)
db1 = sum(dZ1, axis=0)    # (512,)
```

### Update

```
W1 = W1 - lr * dW1
b1 = b1 - lr * db1
W2 = W2 - lr * dW2
b2 = b2 - lr * db2
```

---

## 📞 Contacto y Soporte

- **Reuniones de sincronización**: Semanal (presencial o call)
- **Validación de código**: Antes de cada merge a `main`
- **Dudas técnicas**: Compartir en grupo o consultar con profesor

---

**Última actualización**: 26 de noviembre de 2025  
**Versión**: 1.0  
**Estado**: ✅ Aprobado por ambos miembros del equipo
