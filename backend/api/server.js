import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { loadModel } from './mlp.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Información de los modelos disponibles
const models = {
  sequential: {
    name: 'C Secuencial',
    path: '../c_secuencial/bin/train_seq.exe',
    accuracy: 93.56,
    time: 1539,
    description: 'Implementación secuencial en C puro'
  },
  openmp: {
    name: 'C + OpenMP',
    path: '../c_openmp/bin/train_openmp.exe',
    accuracy: 93.56,
    time: 346,
    threads: 8,
    speedup: 4.45,
    description: 'Implementación paralela con OpenMP (8 threads)'
  }
};

// Resultados de benchmarks
const benchmarks = [
  { version: 'Secuencial', time: 1539, threads: 1, speedup: 1.0, efficiency: 100 },
  { version: 'OpenMP', time: 1593, threads: 1, speedup: 0.97, efficiency: 97 },
  { version: 'OpenMP', time: 873, threads: 2, speedup: 1.76, efficiency: 88 },
  { version: 'OpenMP', time: 526, threads: 4, speedup: 2.93, efficiency: 73 },
  { version: 'OpenMP', time: 346, threads: 8, speedup: 4.45, efficiency: 56 },
];

// Rutas

// GET /api/health - Verificar que el servidor está funcionando
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'MLP MNIST API está funcionando',
    timestamp: new Date().toISOString()
  });
});

// GET /api/models - Obtener lista de modelos disponibles
app.get('/api/models', (req, res) => {
  res.json({
    success: true,
    models: Object.entries(models).map(([id, info]) => ({
      id,
      ...info
    }))
  });
});

// GET /api/benchmarks - Obtener resultados de benchmarks
app.get('/api/benchmarks', (req, res) => {
  res.json({
    success: true,
    data: benchmarks
  });
});

// POST /api/predict - Hacer predicción con un modelo
app.post('/api/predict', async (req, res) => {
  const { model = 'sequential', image } = req.body;

  if (!image || !Array.isArray(image) || image.length !== 784) {
    return res.status(400).json({
      success: false,
      error: 'Imagen inválida. Debe ser un array de 784 valores (28x28)'
    });
  }

  if (!models[model]) {
    return res.status(400).json({
      success: false,
      error: 'Modelo no encontrado. Usa "sequential" o "openmp"'
    });
  }

  try {
    // Cargar modelo real (se cachea después de la primera carga)
    const mlpModel = await loadModel();
    
    // Hacer predicción real con la red neuronal
    const prediction = mlpModel.predict(image);

    res.json({
      success: true,
      model: models[model].name,
      prediction: prediction.digit,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities
    });
  } catch (error) {
    console.error('Error en predicción:', error);
    
    // Si el modelo no está disponible, usar simulación
    const prediction = simulatePredict(image);
    
    res.json({
      success: true,
      model: models[model].name + ' (simulado)',
      prediction: prediction.digit,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      warning: 'Usando predicción simulada. Ejecuta export_weights.exe primero.'
    });
  }
});

// GET /api/dataset/random - Obtener imagen aleatoria del dataset
app.get('/api/dataset/random', (req, res) => {
  const { set = 'test' } = req.query;
  
  // Aquí deberíamos leer del dataset real
  // Por ahora simulamos
  const randomDigit = Math.floor(Math.random() * 10);
  const randomImage = Array(784).fill(0); // Imagen vacía simulada
  
  res.json({
    success: true,
    digit: randomDigit,
    image: randomImage,
    set: set
  });
});

// Función simulada de predicción
// En producción, aquí cargaríamos los pesos del modelo C y haríamos forward pass
function simulatePredict(image) {
  // Detectar si hay píxeles "dibujados" (> 0.5)
  const hasDrawing = image.some(pixel => pixel > 0.5);
  
  if (!hasDrawing) {
    // Si no hay dibujo, devolver probabilidades uniformes
    return {
      digit: -1,
      confidence: 0,
      probabilities: Array(10).fill(0.1)
    };
  }

  // Simulación simple: generar predicción aleatoria pero realista
  const probabilities = Array(10).fill(0).map(() => Math.random() * 0.3);
  const predictedDigit = Math.floor(Math.random() * 10);
  probabilities[predictedDigit] = 0.5 + Math.random() * 0.4; // 50-90% de confianza
  
  // Normalizar
  const sum = probabilities.reduce((a, b) => a + b, 0);
  const normalized = probabilities.map(p => p / sum);
  
  return {
    digit: predictedDigit,
    confidence: normalized[predictedDigit],
    probabilities: normalized
  };
}

// Iniciar servidor
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🧠 MLP MNIST API - Backend Server                    ║
║                                                            ║
║       ✅ Servidor corriendo en http://localhost:${PORT}      ║
║       ✅ Modelos disponibles: Sequential, OpenMP           ║
║                                                            ║
║       Endpoints:                                           ║
║       - GET  /api/health         Estado del servidor       ║
║       - GET  /api/models         Lista de modelos          ║
║       - GET  /api/benchmarks     Resultados de benchmarks  ║
║       - POST /api/predict        Hacer predicción          ║
║       - GET  /api/dataset/random Imagen aleatoria          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
  `);
});

export default app;
