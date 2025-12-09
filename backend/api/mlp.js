// Implementación de MLP en JavaScript para predicciones
// Forward pass usando los pesos exportados del modelo C

class MLP {
  constructor(weights) {
    this.inputSize = weights.input_size;
    this.hiddenSize = weights.hidden_size;
    this.outputSize = weights.output_size;
    
    // Cargar pesos
    this.W1 = weights.W1;
    this.b1 = weights.b1;
    this.W2 = weights.W2;
    this.b2 = weights.b2;
  }

  // Multiplicación de matrices
  matmul(A, B, rowsA, colsA, colsB) {
    const result = new Array(rowsA * colsB);
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += A[i * colsA + k] * B[k * colsB + j];
        }
        result[i * colsB + j] = sum;
      }
    }
    return result;
  }

  // ReLU activation
  relu(x) {
    return x.map(val => Math.max(0, val));
  }

  // Softmax activation
  softmax(x) {
    const max = Math.max(...x);
    const exps = x.map(val => Math.exp(val - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(val => val / sum);
  }

  // Forward pass
  predict(input) {
    // Asegurar que input sea array de 784 valores
    if (input.length !== this.inputSize) {
      throw new Error(`Input debe tener ${this.inputSize} valores, tiene ${input.length}`);
    }

    // Z1 = X @ W1 + b1
    const Z1 = this.matmul(input, this.W1, 1, this.inputSize, this.hiddenSize);
    const Z1_with_bias = Z1.map((val, i) => val + this.b1[i]);
    
    // A1 = ReLU(Z1)
    const A1 = this.relu(Z1_with_bias);
    
    // Z2 = A1 @ W2 + b2
    const Z2 = this.matmul(A1, this.W2, 1, this.hiddenSize, this.outputSize);
    const Z2_with_bias = Z2.map((val, i) => val + this.b2[i]);
    
    // A2 = Softmax(Z2)
    const probabilities = this.softmax(Z2_with_bias);
    
    // Encontrar el dígito con mayor probabilidad
    const prediction = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[prediction];
    
    return {
      digit: prediction,
      confidence: confidence,
      probabilities: probabilities
    };
  }
}

// Cargar modelo específico (sequential o openmp)
const modelCache = {};

export async function loadModel(modelType = 'openmp') {
  if (modelCache[modelType]) {
    return modelCache[modelType];
  }

  try {
    const fs = await import('fs');
    const path = await import('path');
    const { fileURLToPath } = await import('url');
    
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    
    const weightsPath = path.join(__dirname, `model_weights_${modelType}.json`);
    const weightsData = fs.readFileSync(weightsPath, 'utf8');
    const weights = JSON.parse(weightsData);
    
    modelCache[modelType] = new MLP(weights);
    console.log(`✅ Modelo ${modelType.toUpperCase()} cargado exitosamente`);
    console.log(`   - Input: ${weights.input_size}, Hidden: ${weights.hidden_size}, Output: ${weights.output_size}`);
    
    return modelCache[modelType];
  } catch (error) {
    console.error(`❌ Error cargando modelo ${modelType}:`, error.message);
    console.error(`   Asegúrate de haber ejecutado: backend/c_${modelType}/export_weights`);
    return null;
  }
}

export default MLP;
