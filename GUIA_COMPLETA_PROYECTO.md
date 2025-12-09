# 🎓 GUÍA COMPLETA DEL PROYECTO MLP-MNIST CONCURRENTE

**Estudiante**: Carlos Gómez  
**Fecha**: 8 de diciembre de 2025  
**Rama actual**: `dev` (C + OpenMP + Frontend/Backend)  
**Compañero**: Rama `devS` (Python secuencial + multiprocessing)

---

## 📊 RESUMEN EJECUTIVO

### ¿Qué es este proyecto?

Es una implementación **desde cero** de una Red Neuronal MLP (Perceptrón Multicapa) para clasificar dígitos escritos a mano (dataset MNIST). El objetivo **NO es la precisión**, sino **comparar el rendimiento de diferentes paradigmas de programación concurrente**.

### Arquitectura de la Red Neuronal

```
ENTRADA (784 neuronas)  →  OCULTA (512 neuronas, ReLU)  →  SALIDA (10 neuronas, Softmax)
     28x28 píxeles              Aprende patrones              0,1,2,3,4,5,6,7,8,9
```

### Implementaciones Requeridas (6 versiones)

| #   | Versión                | Estado       | Responsable | Speedup Esperado   |
| --- | ---------------------- | ------------ | ----------- | ------------------ |
| 1a  | Python Secuencial      | ✅ En `devS` | Compañero   | Baseline (1.0×)    |
| 1b  | **C Secuencial**       | ✅ **TÚ**    | **TÚ**      | 2-3× vs Python     |
| 2a  | Python Multiprocessing | ✅ En `devS` | Compañero   | 2-4× vs Python seq |
| 2b  | **C + OpenMP**         | ✅ **TÚ**    | **TÚ**      | **4-8× vs C seq**  |
| 3a  | CUDA (C++)             | ⏳ Pendiente | Ambos       | 10-50×             |
| 3b  | PyCUDA (Python)        | ⏳ Pendiente | Ambos       | 8-30×              |

---

## 🏗️ ARQUITECTURA TÉCNICA

### Backend (Tu parte - rama `dev`)

#### 1. **C Secuencial** (`backend/c_secuencial/`)

- **Propósito**: Baseline en C, más rápido que Python pero sin paralelización
- **Componentes**:
  - `include/matrix.h`: Multiplicación de matrices (GEMM)
  - `include/mlp.h`: Forward/Backward propagation
  - `include/data.h`: Carga de archivos `.bin` del dataset
  - `src/train.c`: Loop de entrenamiento (10 epochs)
  - `src/export_weights.c`: Exporta pesos a JSON para el frontend

**Compilar y ejecutar**:

```bash
cd backend/c_secuencial
make
./bin/train_seq.exe
```

**Salida**:

- `backend/results/raw/c_sequential.csv` (métricas por época)
- `backend/api/model_weights_sequential.json` (pesos para frontend)

#### 2. **C + OpenMP** (`backend/c_openmp/`)

- **Propósito**: Paralelización con hilos (memoria compartida)
- **Optimizaciones**:
  - `#pragma omp parallel for` en multiplicación de matrices
  - Paralelización del batch processing
  - Uso de `OMP_NUM_THREADS` para escalar

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

#### 3. **API Node.js** (`backend/api/`)

- **Propósito**: Servidor REST para que el frontend haga predicciones
- **Endpoints**:
  - `GET /api/health`: Verificar servidor
  - `GET /api/models`: Listar modelos disponibles
  - `POST /api/predict`: Predecir dígito

**Iniciar**:

```bash
cd backend/api
npm install
npm start  # Puerto 3001
```

#### 4. **Frontend React** (`frontend/`)

- **Propósito**: Interfaz para dibujar y predecir dígitos
- **Características**:
  - Canvas para dibujar (28×28)
  - Selección de modelo (Sequential/OpenMP)
  - Visualización de probabilidades

**Iniciar**:

```bash
cd frontend
npm install
npm run dev  # Puerto 5173
```

### Dataset (`backend/data/mnist/`)

**Archivos generados por scripts Python**:

- `train_images.bin`: 60,000 imágenes (180 MB)
- `train_labels.bin`: 60,000 etiquetas one-hot (2.4 MB)
- `test_images.bin`: 10,000 imágenes (30 MB)
- `test_labels.bin`: 10,000 etiquetas (0.4 MB)

**⚠️ Estos archivos NO están en el repositorio** (son generados localmente).

---

## 🔄 GESTIÓN DE GIT

### Estado Actual

**Tu rama `dev`**:

- 1 commit adelante de `origin/dev`
- Cambios sin commitear:
  - ✅ Frontend completo
  - ✅ API refactorizada
  - ✅ Exportación de pesos mejorada
  - ⚠️ Archivos eliminados: `CHECKLIST.md`, `RESUMEN.md`, `start.sh`

**Rama `devS` (compañero)**:

- Contiene Python secuencial y multiprocessing
- Movió carpetas `c_*` a la raíz (diferente estructura)
- Eliminó todo el frontend y API

### Plan de Integración

```bash
# PASO 1: Commitear tus cambios actuales
cd "c:\Users\carli\OneDrive\Desktop\Universidad de Caldas\Semestre VII\Concurrentes\Proyecto\mlp-mnist-concurrente"

git add .
git commit -m "feat: Frontend React + API Node.js + exportación de pesos mejorada"

# PASO 2: Pushear tu rama dev
git push origin dev

# PASO 3: Traer los cambios de Python (devS) SIN sobrescribir tu trabajo
# Opción A: Merge (recomendado)
git merge origin/devS -m "merge: Integrar implementaciones Python de devS"

# Si hay conflictos (es probable), Git te avisará
# Los conflictos estarán en archivos que ambos modificaron

# Opción B: Cherry-pick (más control)
# Solo traer los archivos de Python sin tocar tu estructura
git checkout origin/devS -- py_secuencial
git checkout origin/devS -- py_multiprocessing
git commit -m "feat: Agregar implementaciones Python desde devS"

# PASO 4: Verificar estructura final
ls
```

**⚠️ RECOMENDACIÓN**: Usa la **Opción B (cherry-pick)** porque `devS` tiene una estructura diferente (movió carpetas) y podría romper tu frontend/backend.

---

## 📐 MATEMÁTICAS DEL MLP

### Forward Propagation (Predicción)

```
1. Z1 = X @ W1 + b1        # (batch, 512) = (batch, 784) @ (784, 512) + (512,)
2. A1 = ReLU(Z1)           # A1[i] = max(0, Z1[i])
3. Z2 = A1 @ W2 + b2       # (batch, 10) = (batch, 512) @ (512, 10) + (10,)
4. A2 = Softmax(Z2)        # A2[j] = exp(Z2[j]) / sum(exp(Z2))
```

### Backward Propagation (Aprendizaje)

```
1. dZ2 = A2 - Y_true          # (batch, 10) Error en la salida
2. dW2 = A1^T @ dZ2 / batch   # (512, 10) Gradiente de W2
3. db2 = sum(dZ2) / batch     # (10,) Gradiente de b2
4. dA1 = dZ2 @ W2^T           # (batch, 512) Error propagado
5. dZ1 = dA1 ⊙ ReLU'(Z1)     # (batch, 512) ⊙ = elemento a elemento
6. dW1 = X^T @ dZ1 / batch    # (784, 512) Gradiente de W1
7. db1 = sum(dZ1) / batch     # (512,) Gradiente de b1
```

### Actualización de Pesos

```
W1 = W1 - α * dW1   # α = 0.01 (learning rate)
b1 = b1 - α * db1
W2 = W2 - α * dW2
b2 = b2 - α * db2
```

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
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}

// DESPUÉS (Paralelo)
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
```

**Explicación**:

- `collapse(2)`: Combina los 2 loops externos en uno solo (más trabajo paralelo)
- `schedule(dynamic)`: Distribuye trabajo dinámicamente (mejor balanceo)

#### 2. Reducción Paralela para Gradientes

```c
#pragma omp parallel for reduction(+:db2[:OUTPUT_SIZE])
for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        db2[j] += gradients[i * OUTPUT_SIZE + j];
    }
}
```

**Explicación**:

- `reduction(+:array)`: Cada hilo acumula en su copia privada, luego se suman

### Escalabilidad Medida

| Hilos | Tiempo (s) | Speedup   | Eficiencia |
| ----- | ---------- | --------- | ---------- |
| 1     | 1539       | 1.0×      | 100%       |
| 2     | 820        | 1.88×     | 94%        |
| 4     | 450        | 3.42×     | 86%        |
| 8     | 346        | **4.45×** | 56%        |

**Observación**: La eficiencia baja al aumentar hilos (Ley de Amdahl).

---

## 🧪 EXPERIMENTOS Y MÉTRICAS

### Métricas a Recopilar

Para cada implementación, registra en `results/raw/<nombre>.csv`:

```csv
epoch,train_loss,train_accuracy,test_accuracy,time_seconds
1,0.532,0.842,0.838,154.3
2,0.321,0.906,0.901,152.1
...
```

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

#### Diapositiva 9: Conclusiones

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
✅ c_secuencial/     (compilable con make)
✅ c_openmp/         (compilable con make)
⏳ pycuda_gpu/       (pendiente)
✅ py_secuencial/    (en rama devS)
✅ py_multiprocessing/ (en rama devS)
✅ frontend/         (extra, no requerido)
✅ backend/api/      (extra, no requerido)
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

**Lo que YA TIENES (rama `dev`)**:

- ✅ C Secuencial (completo y funcional)
- ✅ C OpenMP (4.45× speedup)
- ✅ Frontend React (extra, no obligatorio)
- ✅ API Node.js (extra, no obligatorio)
- ✅ Sistema de exportación de pesos

**Lo que FALTA (rama `devS` de tu compañero)**:

- Python Secuencial
- Python Multiprocessing

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
