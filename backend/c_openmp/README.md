# MLP-MNIST con OpenMP

Implementación paralela de un perceptrón multicapa para clasificación de dígitos MNIST usando **OpenMP**.

## 📋 Características

- **Paralelización**: OpenMP en operaciones matriciales y entrenamiento
- **Arquitectura**: 784 → 512 (ReLU) → 10 (Softmax)
- **Optimización**: Compilación con `-O3` y `-fopenmp`
- **Escalabilidad**: Configuración dinámica del número de threads

## 🚀 Compilación

### Windows (MSYS2)

```bash
./compile.bat
```

### Linux/Mac

```bash
make
```

## ▶️ Ejecución

### Con número específico de threads:

```bash
# Windows
set OMP_NUM_THREADS=4 && bin\train_openmp.exe

# Linux/Mac
OMP_NUM_THREADS=4 ./bin/train_openmp.exe
```

### Usando Makefile:

```bash
make run THREADS=4
```

## 🧪 Benchmark

Ejecutar con diferentes números de threads:

```bash
make benchmark
```

Esto ejecutará el entrenamiento con 1, 2, 4 y 8 threads automáticamente.

## 📊 Operaciones Paralelizadas

### Operaciones Matriciales

- `matrix_multiply`: Multiplicación de matrices (triple loop con `collapse(2)`)
- `matrix_transpose_multiply`: Multiplicación con transposición
- `matrix_transpose`: Transposición de matrices

### Operaciones Vectoriales

- `matrix_add`: Suma elemento a elemento
- `matrix_subtract`: Resta elemento a elemento
- `matrix_elementwise_multiply`: Producto elemento a elemento
- `matrix_scale`: Escalado por escalar

### Funciones de Activación

- `relu`: Paralelización por elementos
- `relu_derivative`: Derivada de ReLU
- `softmax`: Paralelización por batch

## 📈 Rendimiento Esperado

| Threads | Tiempo (1 época) | Speedup |
| ------- | ---------------- | ------- |
| 1       | ~124s            | 1.0x    |
| 2       | ~85s             | 1.46x   |
| 4       | ~98s             | 1.26x   |
| 8       | ~TBD             | TBD     |

_Nota: Resultados pueden variar según el hardware_

## 📁 Estructura

```
c_openmp/
├── src/
│   ├── data.c       # Carga de dataset
│   ├── matrix.c     # Operaciones matriciales (con OpenMP)
│   ├── mlp.c        # Red neuronal
│   └── train.c      # Loop de entrenamiento
├── include/
│   ├── data.h
│   ├── matrix.h
│   └── mlp.h
├── Makefile         # Sistema de build
├── compile.bat      # Script de compilación Windows
└── README.md        # Este archivo
```

## 🔧 Requisitos

- GCC con soporte OpenMP (incluido en MSYS2 UCRT64)
- Compilador: `gcc -fopenmp`
- Dataset MNIST en formato binario (en `../../data/mnist/`)

## 📝 Notas

- Los resultados se exportan a `../results/raw/c_openmp.csv`
- El speedup depende del número de cores disponibles
- Usar `OMP_NUM_THREADS` para controlar paralelización
- Para mejor rendimiento, usar número de threads = número de cores físicos
