# Backend - MLP MNIST

Backend del proyecto con implementaciones en C (secuencial y OpenMP) para reconocimiento de dígitos manuscritos MNIST.

## 📁 Estructura

```
backend/
├── c_secuencial/       # Implementación secuencial en C
├── c_openmp/           # Implementación paralela con OpenMP
├── data/               # Dataset MNIST (60k train + 10k test)
├── docs/               # Documentación técnica
├── results/            # Resultados de entrenamiento (CSV, pesos)
├── scripts/            # Scripts de procesamiento
└── visualize_mnist.py  # Visualizador de imágenes ASCII
```

## 🚀 Compilar y Ejecutar

### Versión Secuencial

```bash
cd c_secuencial
make
./bin/train_seq.exe
```

### Versión OpenMP

```bash
cd c_openmp
make
export OMP_NUM_THREADS=8
./bin/train_openmp.exe
```

## 📊 Resultados

- **Secuencial**: 93.56% accuracy, 1,539s
- **OpenMP (8 threads)**: 93.56% accuracy, 346s (4.45× speedup)

## 🔗 Frontend

El frontend React se encuentra en `../frontend/`
