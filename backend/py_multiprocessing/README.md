# MLP MNIST en Python con multiprocessing (Paralelismo de Datos)

Implementa un MLP de 3 capas usando NumPy y entrena con paralelismo de datos:
- Divide cada mini-batch en sub-lotes para varios procesos.
- Cada worker calcula gradientes (Fase 3) en su sub-lote.
- El maestro promedia gradientes y actualiza pesos (Fase 4).

## Estructura
- `src/mlp.py`: MLP (fases 1–4) idéntico al secuencial.
- `src/data_loader.py`: Carga MNIST (IDX o BIN).
- `src/train_mp.py`: Entrenamiento con multiprocessing.
- `requirements.txt`: Dependencias.

## Uso
```powershell
# Desde la raíz del proyecto
cd backend/py_multiprocessing

# Crear venv (opcional)
python -m venv .venv; .venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Entrenar con 4 procesos usando IDX
cd src
python train_mp.py --epochs 5 --batch-size 512 --hidden-dim 512 --seed 42 --workers 4

# Entrenar con BIN
python train_mp.py --use-bin --epochs 5 --batch-size 512 --hidden-dim 512 --seed 42 --workers 4
```

## Notas
- Asegúrate de usar `if __name__ == '__main__':` en Windows.
- Ajusta `--workers` según núcleos disponibles.
- El promedio de gradientes se hace por suma en cada sub-lote y normalización por tamaño total del mini-batch.
