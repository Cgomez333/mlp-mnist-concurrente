# MLP MNIST - Implementación en C Secuencial

Esta carpeta contiene la implementación baseline en C puro (sin paralelización) del MLP para MNIST.

## 📁 Estructura

```
c_secuencial/
├── include/
│   ├── data.h      # Carga de datos desde archivos .bin
│   ├── matrix.h    # Operaciones de matrices
│   └── mlp.h       # Estructura y funciones del MLP
├── src/
│   ├── data.c      # Implementación de carga de datos
│   ├── matrix.c    # Implementación de operaciones matriciales
│   ├── mlp.c       # Implementación del MLP (forward, backward, update)
│   └── train.c     # Programa principal de entrenamiento
├── Makefile        # Compilación automatizada
└── README.md       # Este archivo
```

## 🎯 Especificaciones

- **Arquitectura**: 784 → 512 (ReLU) → 10 (Softmax)
- **Loss**: Cross-Entropy
- **Optimizador**: Gradient Descent
- **Hiperparámetros**:
  - Epochs: 10
  - Batch size: 64
  - Learning rate: 0.01
  - Seed: 42

## 🔧 Compilación

### Requisitos

- GCC (versión 7.0 o superior)
- Make
- Librería matemática estándar (libm)

### Compilar

```bash
make
```

Esto genera el ejecutable en `bin/train_seq`.

### Compilar y ejecutar

```bash
make run
```

### Limpiar archivos compilados

```bash
make clean
```

## ▶️ Ejecución

```bash
cd bin
./train_seq
```

O simplemente:

```bash
make run
```

## 📊 Salida Esperada

```
=================================================================
MLP MNIST - Implementación Secuencial en C
=================================================================

✓ Dataset cargado exitosamente
  - 60000 imágenes de 784 características
  - Labels one-hot de 10 clases

MLP creado: 784 -> 512 -> 10 (batch_size=64)

=================================================================
Iniciando entrenamiento (10 epochs, batch_size=64, lr=0.010)
=================================================================

Epoch  1/10 - Loss: 0.4523 - Accuracy: 0.8712 - Time: 12.34s
Epoch  2/10 - Loss: 0.2891 - Accuracy: 0.9124 - Time: 12.21s
...
Epoch 10/10 - Loss: 0.1523 - Accuracy: 0.9512 - Time: 12.18s

=================================================================
Entrenamiento completado en 121.56 segundos
=================================================================

Test Loss: 0.1689 - Test Accuracy: 0.9456

Resultados guardados en: results/raw/c_sequential.csv
```

## 📝 Tareas de Implementación

### ✅ Ya implementado:

- [x] Estructura del proyecto
- [x] Carga de datos (`data.c`)
- [x] Operaciones básicas de matrices (`matrix.c`)
- [x] Inicialización del MLP
- [x] Programa principal de entrenamiento

### 🔧 Por implementar (TÚ):

#### En `mlp.c`:

1. **`mlp_forward()`**:

   - Z1 = X @ W1 + b1
   - A1 = ReLU(Z1)
   - Z2 = A1 @ W2 + b2
   - A2 = Softmax(Z2)

2. **`mlp_backward()`**:

   - dZ2 = A2 - Y
   - dW2 = A1^T @ dZ2
   - db2 = sum(dZ2, axis=0)
   - dA1 = dZ2 @ W2^T
   - dZ1 = dA1 ⊙ ReLU'(Z1)
   - dW1 = X^T @ dZ1
   - db1 = sum(dZ1, axis=0)

3. **`mlp_update_params()`**:
   - W1 -= lr \* dW1 / batch_size
   - b1 -= lr \* db1 / batch_size
   - W2 -= lr \* dW2 / batch_size
   - b2 -= lr \* db2 / batch_size

## 🐛 Debug

### Ver valores intermedios

Descomen los `printf` en `mlp.c` o usa `print_matrix()`:

```c
print_matrix("W1", mlp->W1, 5, 5);  // Primeras 5x5 de W1
print_matrix("A2", mlp->A2, batch_size, 10);  // Predicciones
```

### Verificar gradientes

Temporalmente en `mlp_backward()`:

```c
printf("dW1 sum: %.6f\n", sum_all(mlp->dW1, 784 * 512));
printf("dW2 sum: %.6f\n", sum_all(mlp->dW2, 512 * 10));
```

Si los gradientes son NaN o explotan, revisa:

- Softmax tiene overflow → Usa max normalization (ya está implementado)
- Learning rate muy alto → Reduce a 0.001
- Divisiones por cero → Agrega epsilon (1e-10)

## 📊 Resultados

Los resultados se guardan automáticamente en:

```
../../results/raw/c_sequential.csv
```

Formato:

```csv
implementation,language,parallelization,workers_threads,batch_size,epochs,learning_rate,hidden_neurons,total_time_sec,avg_epoch_time,final_loss,final_accuracy,speedup_vs_baseline,notes
c_seq,c,none,1,64,10,0.010,512,121.56,12.16,0.1689,0.9456,1.00,baseline_c
```

## 🔍 Validación

### Compara con Python

Tu compañero debe ejecutar su versión Python secuencial con el mismo seed (42).

Los resultados deben ser similares (diferencia < 1e-3):

```
Python Final Loss:    0.1692  ← OK
C Final Loss:         0.1689  ← OK

Diferencia: 0.0003 ✓
```

## 🚀 Próximos Pasos

Una vez que esta versión funcione:

1. ✅ Verificar que loss disminuye
2. ✅ Verificar accuracy > 90%
3. ✅ Commit y push
4. 🔄 **Pasar a `c_openmp/`** para paralelizar

## 💡 Tips

- Compila con `-O3` para optimización (ya está en Makefile)
- Usa `valgrind` para detectar memory leaks:
  ```bash
  valgrind --leak-check=full ./bin/train_seq
  ```
- Si es muy lento, reduce epochs o usa subset del dataset

## 📚 Referencias

- Fundamentación matemática: `docs/experiment_design.md`
- Formato de datos: `docs/WORKFLOW.md`
