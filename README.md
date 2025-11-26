# MLP MNIST Concurrente

Implementación y paralelización de una Red Neuronal MLP desde cero para clasificación de dígitos MNIST.

## 📋 Descripción del Proyecto

Este proyecto implementa un **Perceptrón Multicapa (MLP)** desde cero en diferentes paradigmas de programación para comparar su rendimiento:

- **Python Secuencial**: Implementación baseline con NumPy
- **Python Multiprocessing**: Paralelización con procesos
- **C Secuencial**: Implementación optimizada en C
- **C + OpenMP**: Paralelización con memoria compartida
- **PyCUDA**: Aceleración en GPU

**Objetivo**: Analizar cuellos de botella computacionales y medir Speedup, Overhead y Ley de Amdahl.

## 🎯 Especificaciones

### Arquitectura MLP (Fija)

- **Entrada**: 784 neuronas (28×28 píxeles)
- **Capa Oculta**: 512 neuronas (ReLU)
- **Salida**: 10 neuronas (Softmax)
- **Loss**: Cross-Entropy

### Hiperparámetros

```python
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 64  # CPU implementations
RANDOM_SEED = 42
```

## 📁 Estructura del Proyecto

```
mlp-mnist-concurrente/
├── README.md
├── docs/
│   └── experiment_design.md          # Documentación completa
├── data/
│   └── mnist/                        # Dataset MNIST
├── python_secuencial/                # Implementación Python base
├── python_multiprocessing/           # Python paralelo
├── pycuda_gpu/                       # Implementación GPU
├── c_secuencial/                     # Implementación C base
├── c_openmp/                         # C con OpenMP
├── scripts/                          # Scripts de análisis
└── results/                          # Resultados y gráficas
    ├── raw/                          # CSVs con métricas
    │   └── weights/                  # Pesos finales
    └── figures/                      # Gráficas comparativas
```

## 🚀 Inicio Rápido

### Requisitos

**Python**:

```bash
pip install numpy matplotlib
pip install pycuda  # Solo para versión GPU
```

**C/C++**:

```bash
gcc --version  # GCC con soporte OpenMP
```

### Descarga del Dataset

```bash
cd scripts
python download_mnist.py
python preprocess_data.py  # Genera archivos .bin para C
```

## 🔧 Ejecución

### Python Secuencial

```bash
cd python_secuencial
python train.py
```

### Python Multiprocessing

```bash
cd python_multiprocessing
python train_parallel.py --processes 4
```

### C Secuencial

**Primero instala GCC** (ver `docs/INSTALL_C_TOOLS.md`):

```bash
cd c_secuencial

# Opción 1: Con Make (si tienes MinGW/MSYS2)
make

# Opción 2: Script de Windows
compile.bat

# Ejecutar
./bin/train_seq.exe  # Windows
./bin/train_seq      # Linux/Mac
```

### C + OpenMP

```bash
cd c_openmp

# Compilar
make
# o
compile.bat

# Ejecutar con diferentes hilos
set OMP_NUM_THREADS=1 && ./bin/train_omp.exe  # Windows
export OMP_NUM_THREADS=8 && ./bin/train_omp   # Linux/Mac
```

### PyCUDA (en Colab)

```bash
# Ver notebooks/pycuda_experiments.ipynb
```

## 📊 Análisis de Resultados

```bash
cd scripts
python aggregate_results.py  # Consolida todos los CSVs
python plot_results.py        # Genera gráficas
```

## 📈 Métricas Evaluadas

- ⏱️ Tiempo total de entrenamiento (10 epochs)
- 🚀 Speedup vs implementación secuencial
- 📉 Overhead de paralelización
- 📊 Ley de Amdahl
- 🎯 Accuracy y Loss final
- 💾 Transferencia Host↔Device (GPU)

## 📚 Documentación Completa

**📖 Lee [`docs/experiment_design.md`](docs/experiment_design.md)** para:

- ✅ Fundamentación matemática (forward/backward propagation)
- ✅ Detalles de implementación por módulo
- ✅ Protocolo de validación
- ✅ División de responsabilidades
- ✅ Cronograma del proyecto
- ✅ Formato de resultados y CSVs

## ⚠️ Restricciones

**NO se permite**:

- ❌ TensorFlow, Keras, PyTorch
- ❌ Librerías de Deep Learning pre-construidas
- ❌ BLAS/LAPACK en C

**Permitido**:

- ✅ NumPy (solo para Python)
- ✅ OpenMP, CUDA
- ✅ Librerías estándar (stdio, stdlib, math)

## 🎓 Estado del Proyecto

- [x] ✅ Fase 0: Estructura y documentación
- [ ] 🔄 Fase 1: Implementaciones secuenciales
- [ ] ⏳ Fase 2: Paralelización en CPU
- [ ] ⏳ Fase 3: Paralelización en GPU
- [ ] ⏳ Fase 4: Análisis y gráficas
- [ ] ⏳ Fase 5: Informe y presentación

## 👥 Equipo

- **C/OpenMP**: [Tu nombre]
- **Python/PyCUDA**: [Nombre compañero]

## 📄 Proyecto Académico

Universidad de Caldas  
Programación Concurrente y Distribuida - 2025

## 🔗 Enlaces Útiles

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [OpenMP Documentation](https://www.openmp.org/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
