# 📐 Fórmulas de Implementación MLP

Este documento contiene las fórmulas matemáticas exactas que debes implementar en `mlp.c`.

---

## 🔵 Forward Propagation

### Dimensiones

- **X**: (batch_size, 784) = Entrada
- **W1**: (784, 512) = Pesos capa oculta
- **b1**: (512,) = Bias capa oculta
- **W2**: (512, 10) = Pesos capa salida
- **b2**: (10,) = Bias capa salida

### Ecuaciones

```
Z1 = X @ W1 + b1              (batch_size, 512)
A1 = ReLU(Z1)                 (batch_size, 512)
Z2 = A1 @ W2 + b2             (batch_size, 10)
A2 = Softmax(Z2)              (batch_size, 10)
```

### En Código (batch_size = 1)

```c
void mlp_forward(MLP* mlp, float* input, float* output) {
    // 1. Hidden layer: Z1 = X @ W1 + b1
    matrix_multiply(input, mlp->W1, mlp->Z1, 1, 784, 512);
    for (int j = 0; j < 512; j++) {
        mlp->Z1[j] += mlp->b1[j];
    }

    // 2. ReLU: A1 = max(0, Z1)
    relu(mlp->Z1, mlp->A1, 512);

    // 3. Output layer: Z2 = A1 @ W2 + b2
    matrix_multiply(mlp->A1, mlp->W2, mlp->Z2, 1, 512, 10);
    for (int j = 0; j < 10; j++) {
        mlp->Z2[j] += mlp->b2[j];
    }

    // 4. Softmax: A2[i] = exp(Z2[i]) / sum(exp(Z2))
    softmax(mlp->Z2, mlp->A2, 10);

    memcpy(output, mlp->A2, 10 * sizeof(float));
}
```

---

## 🔴 Backward Propagation (Gradientes)

### Ecuaciones (para batch_size = 1)

```
dZ2 = A2 - y_true                    (1, 10)
dW2 = A1^T @ dZ2                     (512, 10)
db2 = dZ2                            (10,)

dA1 = dZ2 @ W2^T                     (1, 512)
dZ1 = dA1 ⊙ ReLU'(Z1)               (1, 512)
dW1 = X^T @ dZ1                      (784, 512)
db1 = dZ1                            (512,)
```

**Donde**:

- `⊙` = multiplicación elemento a elemento (Hadamard)
- `ReLU'(x) = 1 if x > 0, else 0`

### En Código

```c
void mlp_backward(MLP* mlp, float* input, float* target) {
    // 1. Output layer delta: dZ2 = A2 - y_true
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        mlp->dZ2[j] = mlp->A2[j] - target[j];
    }

    // 2. dW2 = A1^T @ dZ2 (outer product)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            mlp->dW2[i * OUTPUT_SIZE + j] = mlp->A1[i] * mlp->dZ2[j];
        }
    }

    // 3. db2 = dZ2 (batch_size = 1, no sum needed)
    memcpy(mlp->db2, mlp->dZ2, OUTPUT_SIZE * sizeof(float));

    // 4. dA1 = dZ2 @ W2^T
    float dA1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            dA1[i] += mlp->dZ2[j] * mlp->W2[i * OUTPUT_SIZE + j];
        }
    }

    // 5. dZ1 = dA1 * ReLU'(Z1)
    relu_derivative(mlp->Z1, mlp->dZ1, HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp->dZ1[i] *= dA1[i];  // Hadamard product
    }

    // 6. dW1 = X^T @ dZ1
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->dW1[i * HIDDEN_SIZE + j] = input[i] * mlp->dZ1[j];
        }
    }

    // 7. db1 = dZ1
    memcpy(mlp->db1, mlp->dZ1, HIDDEN_SIZE * sizeof(float));
}
```

---

## ⚙️ Parameter Update (Gradient Descent)

### Ecuación

```
W = W - η * dW
b = b - η * db
```

**Donde**: η (eta) = learning_rate = 0.01

### En Código

```c
void mlp_update_params(MLP* mlp, float learning_rate) {
    // Update W1
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        mlp->W1[i] -= learning_rate * mlp->dW1[i];
    }

    // Update b1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp->b1[i] -= learning_rate * mlp->db1[i];
    }

    // Update W2
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        mlp->W2[i] -= learning_rate * mlp->dW2[i];
    }

    // Update b2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        mlp->b2[i] -= learning_rate * mlp->db2[i];
    }
}
```

---

## 📊 Loss Function (Cross-Entropy)

### Ecuación

```
L = -∑(y_true[i] * log(y_pred[i]))
```

Para clasificación (one-hot encoding):

```
L = -log(y_pred[correct_class])
```

### En Código (ya implementado en mlp.c)

```c
float mlp_compute_loss(MLP* mlp, float* target) {
    float loss = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (target[i] > 0.5f) {  // One-hot position
            loss = -logf(mlp->A2[i] + 1e-8f);  // Add epsilon for stability
            break;
        }
    }
    return loss;
}
```

---

## 🎯 Activaciones

### ReLU (ya implementado en matrix.c)

```c
// Forward
y = max(0, x)

// Backward (derivative)
dy/dx = 1 if x > 0, else 0
```

### Softmax (ya implementado en matrix.c)

```c
// Forward
y[i] = exp(x[i]) / ∑(exp(x[j]))

// Backward (implicit in dZ2 = A2 - y_true)
// Cross-entropy + Softmax derivative simplifies to: A2 - y_true
```

---

## 🧮 Dimensiones y Almacenamiento

### Matrices en Memoria (Row-Major)

```c
// Matrix (m, n) stored as 1D array
float* matrix = malloc(m * n * sizeof(float));

// Access: matrix[i][j] → matrix[i * n + j]

// Example: W1 (784, 512)
W1[i][j] = W1[i * 512 + j]  // i ∈ [0,783], j ∈ [0,511]
```

### Verificación de Dimensiones

```
Forward:
- X:  (1, 784)
- W1: (784, 512) → Z1: (1, 512)
- A1: (1, 512)
- W2: (512, 10)  → Z2: (1, 10)
- A2: (1, 10)

Backward:
- dZ2: (1, 10)
- dW2: (512, 10)  ← A1^T: (512, 1) @ dZ2: (1, 10)
- dA1: (1, 512)   ← dZ2: (1, 10) @ W2^T: (10, 512)
- dZ1: (1, 512)
- dW1: (784, 512) ← X^T: (784, 1) @ dZ1: (1, 512)
```

---

## 🔍 Debugging Tips

### Imprimir Valores Intermedios

```c
// En mlp_forward()
printf("Z1[0] = %.4f, A1[0] = %.4f\n", mlp->Z1[0], mlp->A1[0]);
printf("Z2 = [%.4f, %.4f, ..., %.4f]\n", mlp->Z2[0], mlp->Z2[1], mlp->Z2[9]);
printf("A2 sum = %.4f (should be 1.0)\n", sum_array(mlp->A2, 10));

// En mlp_backward()
printf("dZ2[0] = %.4f, dW2[0] = %.6f\n", mlp->dZ2[0], mlp->dW2[0]);
```

### Validaciones

```c
// Softmax output suma 1
float sum = 0.0f;
for (int i = 0; i < 10; i++) sum += mlp->A2[i];
assert(fabsf(sum - 1.0f) < 1e-5);

// No NaN
assert(!isnan(mlp->A2[0]));

// No Inf
assert(!isinf(mlp->W1[0]));
```

---

## 📈 Valores Esperados

### Epoch 1

- Loss inicial: ~2.3
- Accuracy: 10-30%

### Epoch 5

- Loss: ~0.8
- Accuracy: 70-85%

### Epoch 10

- Loss: <0.5
- Accuracy: >90%

**Si no converge**:

1. Imprime gradientes (deben ser pequeños, ~1e-3)
2. Verifica que ReLU'(Z1) no sea todo ceros
3. Revisa el signo en update (debe ser `-=`, no `+=`)

---

## 🔗 Referencias

- **experiment_design.md**: Especificación completa
- **matrix.h/matrix.c**: Funciones auxiliares ya implementadas
- **train.c**: Loop de entrenamiento

---

**🚀 Con estas fórmulas puedes completar `mlp.c` directamente!**
