# Dataset MNIST

⚠️ **Los archivos binarios NO están en Git** (son muy grandes: 212 MB total).

## 🔽 Cómo Obtener los Datos

### Opción 1: Scripts Automáticos (Recomendado)

```bash
cd backend/scripts
python download_mnist.py        # Descarga archivos originales (.idx-ubyte)
python preprocess_for_c.py      # Genera archivos .bin para C
```

### Opción 2: Descarga Manual

1. Descargar de http://yann.lecun.com/exdb/mnist/
   - `train-images-idx3-ubyte.gz` (9.9 MB)
   - `train-labels-idx1-ubyte.gz` (28 KB)
   - `t10k-images-idx3-ubyte.gz` (1.6 MB)
   - `t10k-labels-idx1-ubyte.gz` (4 KB)

2. Descomprimir en `backend/data/mnist/`

3. Ejecutar `python backend/scripts/preprocess_for_c.py`

## 📦 Archivos Esperados

Después de ejecutar los scripts, deberías tener:

```
backend/data/mnist/
├── train-images-idx3-ubyte    # Original (47 MB)
├── train-labels-idx1-ubyte    # Original (60 KB)
├── t10k-images-idx3-ubyte     # Original (7.8 MB)
├── t10k-labels-idx1-ubyte     # Original (10 KB)
├── train_images.bin           # Para C (180 MB) - float32
├── train_labels.bin           # Para C (2.4 MB) - float32
├── test_images.bin            # Para C (30 MB) - float32
└── test_labels.bin            # Para C (0.4 MB) - float32
```

## ✅ Verificación

```bash
# Desde Python
python -c "
import os
files = ['train_images.bin', 'train_labels.bin', 'test_images.bin', 'test_labels.bin']
path = 'backend/data/mnist/'
for f in files:
    full_path = os.path.join(path, f)
    if os.path.exists(full_path):
        size_mb = os.path.getsize(full_path) / (1024**2)
        print(f'✓ {f}: {size_mb:.1f} MB')
    else:
        print(f'✗ {f}: FALTA')
"
```

## 🔒 Por Qué No Están en Git

- Los archivos `.bin` pesan **212 MB** en total
- GitHub limita archivos a 100 MB
- No tiene sentido versionarlos (son datos estáticos)
- Cada desarrollador debe generarlos localmente

## 📝 Notas

- Los archivos `.bin` están en `.gitignore`
- Si clonas el repo, ejecuta los scripts antes de compilar C
- El preprocesamiento toma ~30 segundos
