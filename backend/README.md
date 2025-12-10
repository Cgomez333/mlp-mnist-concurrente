# Backend - MLP MNIST

Backend del proyecto con implementaciones en C (secuencial y OpenMP) para reconocimiento de dígitos manuscritos MNIST.

## 📁 Estructura

```
backend/
├── py_secuencial/         # ✅ Implementación Python baseline
├── py_multiprocessing/    # ✅ Implementación Python paralela
├── c_secuencial/          # ✅ Implementación C secuencial
├── c_openmp/              # ✅ Implementación C + OpenMP
├── pycuda_gpu/            # ⏳ Implementación GPU (pendiente)
├── api/                   # Node.js REST API para predicciones
├── data/                  # Dataset MNIST (60k train + 10k test)
├── docs/                  # Documentación técnica
├── results/               # Resultados de entrenamiento (CSV, pesos)
├── scripts/               # Scripts de procesamiento
└── visualize_mnist.py     # Visualizador de imágenes ASCII
```

## 🚀 Compilar y Ejecutar

### Python Secuencial

```bash
cd py_secuencial/src
python train.py --epochs 10 --batch-size 256
```

### Python Multiprocessing

```bash
cd py_multiprocessing/src
python train_mp.py --epochs 10 --workers 4
```

### C Secuencial

```bash
cd c_secuencial
make
./bin/train_seq.exe
```

### C + OpenMP

```bash
cd c_openmp
make
set OMP_NUM_THREADS=8  # Windows
export OMP_NUM_THREADS=8  # Linux/Mac
./bin/train_openmp.exe
```

## 📊 Resultados

- **Python Secuencial**: ~93% accuracy, ~1,800s
- **Python Multiprocessing (4 workers)**: ~93% accuracy, ~900s (2× speedup)
- **C Secuencial**: 93.56% accuracy, 1,539s
- **C OpenMP (8 threads)**: 93.56% accuracy, 346s (4.45× speedup)

## 🔗 Frontend

El frontend React se encuentra en `../frontend/`
