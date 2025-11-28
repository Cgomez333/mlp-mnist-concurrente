# 🔄 Flujo de Trabajo y Dependencias entre Implementaciones

## ⚠️ IMPORTANTE: Orden de Ejecución

### 1️⃣ PRIMERO - Compañero Python (BLOQUEANTE para C)

```bash
cd scripts

# Paso 1: Descargar MNIST
python download_mnist.py
# Genera: data/mnist/*.idx*-ubyte (archivos raw)

# Paso 2: Preprocesar para C
python preprocess_for_c.py
# Genera: data/mnist/*.bin (archivos que C puede leer)
```

**✅ Resultado esperado**:

```
data/mnist/
├── train_images.bin      # (60000, 784) float32 - 180 MB
├── train_labels.bin      # (60000, 10) float32 - 2.4 MB
├── test_images.bin       # (10000, 784) float32 - 30 MB
└── test_labels.bin       # (10000, 10) float32 - 0.4 MB
```

**🚫 SIN ESTOS ARCHIVOS, C NO PUEDE FUNCIONAR**

---

### 2️⃣ SEGUNDO - Ambos pueden trabajar en paralelo

Una vez generados los `.bin`, cada uno puede desarrollar independientemente:

**Compañero Python**:

```bash
cd python_secuencial
python train.py
```

**Tú (C)**:

```bash
cd c_secuencial
make
./train_seq
```

---

## 📦 ¿Qué archivos se comparten?

### Del compañero hacia ti (Python → C)

| Archivo            | Tamaño | Descripción            | Cuándo   |
| ------------------ | ------ | ---------------------- | -------- |
| `train_images.bin` | 180 MB | Imágenes entrenamiento | Fase 0/1 |
| `train_labels.bin` | 2.4 MB | Labels entrenamiento   | Fase 0/1 |
| `test_images.bin`  | 30 MB  | Imágenes prueba        | Fase 0/1 |
| `test_labels.bin`  | 0.4 MB | Labels prueba          | Fase 0/1 |

**Formato**: Arrays contiguos de `float32` (little-endian)

### De ti hacia el compañero (C → Python)

| Archivo            | Tamaño | Descripción             | Cuándo |
| ------------------ | ------ | ----------------------- | ------ |
| `c_sequential.csv` | ~1 KB  | Resultados C secuencial | Fase 1 |
| `c_openmp.csv`     | ~5 KB  | Resultados C OpenMP     | Fase 2 |

**Formato**: CSV con columnas definidas en `experiment_design.md`

---

## 🔍 Cómo verificar que los archivos están correctos

### Desde Python (tu compañero)

```python
import numpy as np
import os

# Verificar train_images.bin
train_imgs = np.fromfile('data/mnist/train_images.bin', dtype=np.float32)
print(f"Train images: {train_imgs.shape} elementos")
print(f"Esperado: {60000 * 784} elementos")
print(f"Rango valores: [{train_imgs.min()}, {train_imgs.max()}]")
print(f"Esperado rango: [0.0, 1.0]")

# Verificar train_labels.bin
train_lbls = np.fromfile('data/mnist/train_labels.bin', dtype=np.float32)
train_lbls = train_lbls.reshape(60000, 10)
print(f"\nTrain labels: {train_lbls.shape}")
print(f"Primera label: {train_lbls[0]}")  # Debe tener un 1.0 y nueve 0.0
```

### Desde C (tú)

```c
// En tu main de prueba
Dataset *dataset = load_dataset("../data/mnist/train_images.bin",
                                "../data/mnist/train_labels.bin",
                                60000);

if (dataset) {
    print_dataset_info(dataset);
    free_dataset(dataset);
} else {
    printf("ERROR: No se pudieron cargar los datos\n");
}
```

---

## 📋 Checklist antes de empezar a programar

### Para tu compañero (Python):

- [ ] Ejecutar `download_mnist.py`
- [ ] Ejecutar `preprocess_for_c.py`
- [ ] Verificar que los 4 archivos `.bin` existan
- [ ] Verificar tamaños de archivos (ver arriba)
- [ ] **COMMIT y PUSH** los archivos `.bin` O compartirlos por Google Drive
  - ⚠️ Nota: Los `.bin` están en `.gitignore` por ser grandes
  - Opción 1: Quitarlos del `.gitignore` temporalmente
  - Opción 2: Compartir por Google Drive/OneDrive
  - Opción 3: Ambos ejecutan los scripts (recomendado)

### Para ti (C):

- [ ] Verificar que `data/mnist/*.bin` existan
- [ ] Compilar `data.c` de prueba
- [ ] Cargar dataset y verificar que no haya errores
- [ ] Verificar que los valores estén en [0, 1]
- [ ] Verificar que labels sean one-hot (un 1.0, nueve 0.0)

---

## 🚀 Recomendación: Ambos ejecuten los scripts

**Para evitar problemas de compatibilidad**:

```bash
# Ambos ejecutan (5 minutos):
cd scripts
python download_mnist.py      # ~50 MB descarga
python preprocess_for_c.py    # Genera los .bin

# Los .bin NO se suben a git (muy grandes)
# Cada uno los genera localmente
```

**Ventajas**:

- ✅ No dependen de transferir archivos grandes
- ✅ Garantiza compatibilidad entre sistemas
- ✅ Reproducible en cualquier máquina

---

## 🔧 Si hay problemas

### "No se encuentra train_images.bin"

```bash
# Verificar que existe
ls -lh data/mnist/

# Si no está, ejecutar:
cd scripts
python preprocess_for_c.py
```

### "Los valores están fuera de rango [0,1]"

```bash
# Regenerar archivos:
cd scripts
rm ../data/mnist/*.bin
python preprocess_for_c.py
```

### "Fread devuelve menos elementos de los esperados"

Puede ser problema de rutas. Desde `c_secuencial/src/train.c`:

```c
// Usar ruta relativa correcta
Dataset *train = load_dataset("../../data/mnist/train_images.bin",
                              "../../data/mnist/train_labels.bin",
                              60000);
```

O usar ruta absoluta temporalmente para debug.

---

## 📞 Comunicación entre ustedes

### Antes de iniciar Fase 1:

**Compañero Python**:

> "Hola, ya ejecuté `download_mnist.py` y `preprocess_for_c.py`.
> Los archivos .bin están en `data/mnist/`.
> ¿Los ejecutas tú también o te los paso por Drive?"

**Tú**:

> "Los ejecuto yo también, más fácil. Listo, confirmado que tengo los 4 .bin"

### Antes de iniciar Fase 4 (análisis):

**Tú**:

> "Ya tengo `c_sequential.csv` y `c_openmp.csv` en `results/raw/`.
> Commitié y pusheé a dev"

**Compañero**:

> "Perfecto, ya puedo ejecutar `aggregate_results.py` y generar las gráficas"

---

## 📊 Estado actual del proyecto

- [x] ✅ Estructura creada
- [x] ✅ Scripts de descarga listos
- [x] ✅ Scripts de preprocesamiento listos
- [x] ✅ Headers de C para leer datos
- [ ] ⏳ **SIGUIENTE**: Ambos ejecutan scripts y verifican datos
- [ ] ⏳ Implementación Python secuencial
- [ ] ⏳ Implementación C secuencial

---

**Última actualización**: 26 nov 2025
