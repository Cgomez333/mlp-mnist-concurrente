# API Backend - MLP MNIST

Servidor REST que expone los modelos MLP entrenados para que el frontend pueda hacer predicciones.

## 🚀 Instalación

```bash
cd backend/api
npm install
```

## ▶️ Ejecutar

```bash
npm start
# o con hot reload
npm run dev
```

Servidor en: **http://localhost:3001**

## 📡 Endpoints

### `GET /api/health`

Verifica que el servidor esté funcionando.

**Respuesta:**

```json
{
  "status": "ok",
  "message": "MLP MNIST API está funcionando",
  "timestamp": "2025-11-27T19:00:00.000Z"
}
```

### `GET /api/models`

Lista los modelos disponibles.

**Respuesta:**

```json
{
  "success": true,
  "models": [
    {
      "id": "sequential",
      "name": "C Secuencial",
      "accuracy": 93.56,
      "time": 1539
    },
    {
      "id": "openmp",
      "name": "C + OpenMP",
      "accuracy": 93.56,
      "time": 346,
      "threads": 8,
      "speedup": 4.45
    }
  ]
}
```

### `GET /api/benchmarks`

Obtiene los resultados de benchmarks completos.

**Respuesta:**

```json
{
  "success": true,
  "data": [
    {
      "version": "Secuencial",
      "time": 1539,
      "threads": 1,
      "speedup": 1.0,
      "efficiency": 100
    },
    {
      "version": "OpenMP",
      "time": 346,
      "threads": 8,
      "speedup": 4.45,
      "efficiency": 56
    }
  ]
}
```

### `POST /api/predict`

Hace una predicción sobre una imagen dibujada.

**Request:**

```json
{
  "model": "sequential",  // o "openmp"
  "image": [0.0, 0.0, ... 784 valores entre 0.0 y 1.0]
}
```

**Respuesta:**

```json
{
  "success": true,
  "model": "C Secuencial",
  "prediction": 5,
  "confidence": 0.89,
  "probabilities": [0.01, 0.02, 0.01, 0.01, 0.01, 0.89, 0.02, 0.01, 0.02, 0.01]
}
```

### `GET /api/dataset/random?set=test`

Obtiene una imagen aleatoria del dataset.

**Respuesta:**

```json
{
  "success": true,
  "digit": 7,
  "image": [0.0, 0.0, ... 784 valores],
  "set": "test"
}
```

## 🔧 Tecnologías

- **Express.js** - Framework web
- **CORS** - Permitir peticiones desde frontend
- **Node.js 18+** - Runtime

## 📝 Notas

Actualmente las predicciones son **simuladas** porque cargar los pesos del modelo C requiere:

1. Exportar pesos entrenados a formato JSON/binario
2. Implementar forward pass en JavaScript
3. O llamar al ejecutable C desde Node.js con `child_process`

Para producción, se recomienda:

- Reentrenar con Python y exportar modelo a ONNX
- O usar WebAssembly para correr el modelo C en el navegador
