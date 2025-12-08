# MLP MNIST en Python (Secuencial con NumPy)

Este módulo implementa un MLP de 3 capas usando únicamente NumPy.
Cubre las fases:
- Fase 1: Forward Propagation
- Fase 2: Cálculo de la Pérdida (cross-entropy)
- Fase 3: Backward Propagation
- Fase 4: Actualización de Pesos (SGD)

## Estructura
- `mlp.py`: Definición del modelo y fases A–D.
- `data_loader.py`: Carga MNIST preprocesado a `.bin` o directamente IDX.
- `train.py`: Script de entrenamiento y evaluación.
- `requirements.txt`: Dependencias mínimas.

## Datos
Usa los archivos MNIST del repo en `data/mnist/`.
Puedes usar directamente los IDX (`train-images-idx3-ubyte`, etc.) o los `.bin` si ya ejecutaste `scripts/preprocess_for_c.py`.

## Uso
```powershell
# Crear venv (opcional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r py_secuencial/requirements.txt

# Entrenar (usando IDX)
python py_secuencial/train.py --data-root data/mnist --epochs 5 --batch-size 256 --hidden-dim 512 --seed 42

# Entrenar (usando BIN generados por preprocess_for_c.py)
python py_secuencial/train.py --data-root data/mnist --use-bin --epochs 5 --batch-size 256 --hidden-dim 512 --seed 42
```

## Resultados
El script imprime pérdida y accuracy por época, junto con tiempo total.
Puedes comparar contra `c_secuencial` y `c_openmp` para calcular speedup.
