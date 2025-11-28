# 🚦 Pasos Siguientes - Implementación C

## ✅ Estado Actual

**Completado**:

- ✅ Estructura del proyecto creada
- ✅ Documentación completa (experiment_design.md, WORKFLOW.md)
- ✅ Dataset MNIST descargado y preprocesado (213 MB en data/mnist/)
- ✅ Archivos .bin generados para C
- ✅ Estructura completa C secuencial creada
- ✅ Implementaciones completas:
  - `matrix.c` - Operaciones matriciales (GEMM, ReLU, Softmax)
  - `data.c` - Carga de dataset desde .bin
  - `train.c` - Loop de entrenamiento con timing
- ✅ Scripts de compilación (Makefile, compile.bat)

**Pendiente**:

- ⏳ Instalar GCC en Windows
- ⏳ Implementar funciones core de MLP (mlp.c)
- ⏳ Compilar y probar C secuencial
- ⏳ Crear versión C+OpenMP
- ⏳ Ejecutar experimentos con diferentes hilos

---

## 📝 Paso 1: Instalar Herramientas de Desarrollo

### Opción Recomendada: MSYS2

1. **Descarga MSYS2**: https://www.msys2.org/
2. **Instala** ejecutando el .exe
3. **Abre MSYS2 terminal** y ejecuta:

```bash
pacman -Syu                          # Actualizar sistema
pacman -S mingw-w64-x86_64-gcc       # Instalar GCC
pacman -S make                        # Instalar Make
```

4. **Agrega a PATH** (Variables de entorno de Windows):

```
C:\msys64\mingw64\bin
```

5. **Verifica** (en terminal VSCode bash):

```bash
gcc --version
make --version
```

**Salida esperada**:

```
gcc (Rev10, Built by MSYS2 project) 13.2.0
GNU Make 4.4.1
```

---

## 📝 Paso 2: Compilar C Secuencial

Una vez instalado GCC:

```bash
cd c_secuencial
compile.bat
```

**Salida esperada**:

```
============================================
 MLP MNIST - Compilacion C Secuencial
============================================

[1/5] Compilando data.c...
[2/5] Compilando matrix.c...
[3/5] Compilando mlp.c...
[4/5] Compilando train.c...
[5/5] Enlazando...

============================================
 COMPILACION EXITOSA
============================================

Ejecutable: bin\train_seq.exe
```

**Si hay errores**: Los TODOs en `mlp.c` causarán warnings pero NO errores de compilación (son comentarios).

---

## 📝 Paso 3: Implementar Funciones Core (mlp.c)

Abre `c_secuencial/src/mlp.c` y completa los TODOs:

### 3.1. `mlp_forward()` - Propagación hacia adelante

```c
void mlp_forward(MLP* mlp, float* input, float* output) {
    // TODO 1: Hidden layer (Z1 = X * W1 + b1)
    matrix_multiply(input, mlp->W1, mlp->Z1, 1, INPUT_SIZE, HIDDEN_SIZE);

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        mlp->Z1[j] += mlp->b1[j];
    }

    // TODO 2: ReLU activation (A1 = ReLU(Z1))
    relu(mlp->Z1, mlp->A1, HIDDEN_SIZE);

    // TODO 3: Output layer (Z2 = A1 * W2 + b2)
    matrix_multiply(mlp->A1, mlp->W2, mlp->Z2, 1, HIDDEN_SIZE, OUTPUT_SIZE);

    for (int j = 0; j < OUTPUT_SIZE; j++) {
        mlp->Z2[j] += mlp->b2[j];
    }

    // TODO 4: Softmax activation (A2 = Softmax(Z2))
    softmax(mlp->Z2, mlp->A2, OUTPUT_SIZE);

    // Copy to output
    memcpy(output, mlp->A2, OUTPUT_SIZE * sizeof(float));
}
```

### 3.2. `mlp_backward()` - Backpropagation

```c
void mlp_backward(MLP* mlp, float* input, float* target) {
    // TODO 1: Output layer delta (dZ2 = A2 - y_true)
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        mlp->dZ2[j] = mlp->A2[j] - target[j];
    }

    // TODO 2: Hidden→Output weights gradient (dW2 = A1^T @ dZ2)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            mlp->dW2[i * OUTPUT_SIZE + j] = mlp->A1[i] * mlp->dZ2[j];
        }
    }

    // TODO 3: Hidden→Output bias gradient (db2 = sum(dZ2))
    memcpy(mlp->db2, mlp->dZ2, OUTPUT_SIZE * sizeof(float));

    // TODO 4: Hidden layer delta (dA1 = dZ2 @ W2^T)
    float dA1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            dA1[i] += mlp->dZ2[j] * mlp->W2[i * OUTPUT_SIZE + j];
        }
    }

    // TODO 5: Apply ReLU derivative (dZ1 = dA1 * relu'(Z1))
    relu_derivative(mlp->Z1, mlp->dZ1, HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp->dZ1[i] *= dA1[i];
    }

    // TODO 6: Input→Hidden weights gradient (dW1 = X^T @ dZ1)
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->dW1[i * HIDDEN_SIZE + j] = input[i] * mlp->dZ1[j];
        }
    }

    // TODO 7: Input→Hidden bias gradient (db1 = sum(dZ1))
    memcpy(mlp->db1, mlp->dZ1, HIDDEN_SIZE * sizeof(float));
}
```

### 3.3. `mlp_update_params()` - Actualización de pesos

```c
void mlp_update_params(MLP* mlp, float learning_rate) {
    // TODO 1: Update W1 (W1 -= lr * dW1)
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        mlp->W1[i] -= learning_rate * mlp->dW1[i];
    }

    // TODO 2: Update b1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp->b1[i] -= learning_rate * mlp->db1[i];
    }

    // TODO 3: Update W2
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        mlp->W2[i] -= learning_rate * mlp->dW2[i];
    }

    // TODO 4: Update b2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        mlp->b2[i] -= learning_rate * mlp->db2[i];
    }
}
```

---

## 📝 Paso 4: Compilar y Probar

```bash
cd c_secuencial
compile.bat
```

Si compila sin errores:

```bash
./bin/train_seq.exe
```

**Salida esperada** (primeras líneas):

```
============================================
  MLP MNIST - Entrenamiento Secuencial
============================================

Epoch 1/10 | Batch 100/937 | Loss: 2.1234 | Acc: 15.3% | Time: 2.34s
Epoch 1/10 | Batch 200/937 | Loss: 1.8456 | Acc: 32.1% | Time: 4.67s
...
Epoch 1/10 | Loss: 1.2345 | Acc: 65.2% | Time: 45.6s
Epoch 2/10 | Loss: 0.8234 | Acc: 78.9% | Time: 43.2s
...
Epoch 10/10 | Loss: 0.3456 | Acc: 92.1% | Time: 41.8s

============================================
  ENTRENAMIENTO COMPLETADO
============================================

Total time: 437.2s
Final accuracy: 92.1%
Results saved: ../results/raw/c_sequential.csv
```

**Validación**:

- ✅ Loss debe bajar de ~2.3 → <0.5
- ✅ Accuracy debe subir a >90%
- ✅ CSV generado en `results/raw/c_sequential.csv`

---

## 📝 Paso 5: Crear Versión OpenMP

1. **Copiar estructura**:

```bash
cp -r c_secuencial/include c_openmp/include
cp -r c_secuencial/src c_openmp/src
```

2. **Editar `c_openmp/src/matrix.c`** - Agregar OpenMP:

```c
// En matrix_multiply()
void matrix_multiply(float* A, float* B, float* C,
                     int m, int n, int k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

// En relu()
void relu(float* input, float* output, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}
```

3. **Compilar OpenMP**:

```bash
cd c_openmp
compile.bat
```

4. **Ejecutar con diferentes hilos**:

```bash
# 1 hilo (baseline)
set OMP_NUM_THREADS=1 && ./bin/train_omp.exe

# 2 hilos
set OMP_NUM_THREADS=2 && ./bin/train_omp.exe

# 4 hilos
set OMP_NUM_THREADS=4 && ./bin/train_omp.exe

# 8 hilos
set OMP_NUM_THREADS=8 && ./bin/train_omp.exe
```

---

## 📝 Paso 6: Analizar Resultados

```bash
cd scripts
python aggregate_results.py
python plot_results.py
```

**Verifica**:

- Speedup vs secuencial
- Overhead de paralelización
- Ley de Amdahl

---

## 🎯 Checklist Final

Antes de considerar completo:

- [ ] GCC instalado y funcionando
- [ ] `c_secuencial/bin/train_seq.exe` ejecuta sin errores
- [ ] Accuracy final >90%
- [ ] CSV generado correctamente
- [ ] `c_openmp/bin/train_omp.exe` compila
- [ ] Experimentos con 1,2,4,8 hilos completados
- [ ] Resultados agregados y graficados
- [ ] Código pusheado a GitHub (rama: dev)

---

## 📚 Referencias Rápidas

- **Instalación GCC**: `docs/INSTALL_C_TOOLS.md`
- **Arquitectura MLP**: `docs/experiment_design.md`
- **Workflow**: `docs/WORKFLOW.md`
- **README C Secuencial**: `c_secuencial/README.md`
- **Makefile**: `c_secuencial/Makefile`

---

## ⚠️ Troubleshooting

### Compilación falla

```bash
gcc --version  # Verifica instalación
which gcc      # Verifica PATH
```

### "Undefined reference to 'exp'"

```bash
# Agrega -lm al enlazar
gcc ... -lm -o bin/train_seq.exe
```

### Accuracy muy baja (<50%)

- Revisa la implementación de `mlp_backward()`
- Verifica que los gradientes se calculen correctamente
- Imprime valores de loss epoch por epoch

### Programa cuelga

- Revisa bucles infinitos
- Verifica accesos a memoria (usar gdb o valgrind)

---

**🚀 ¡Comienza por Paso 1: Instalar GCC!**
