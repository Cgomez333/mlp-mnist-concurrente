/**
 * mlp.c - Implementación de red neuronal MLP
 *
 * Multi-Layer Perceptron para clasificación de dígitos MNIST
 * Arquitectura: 784 -> 512 (ReLU) -> 10 (Softmax)
 * Optimizador: Gradient Descent
 */

#include "mlp.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

MLP *mlp_create(size_t input_size, size_t hidden_size, size_t output_size,
                size_t batch_size, unsigned int seed)
{
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    if (!mlp)
        return NULL;

    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;
    mlp->batch_size = batch_size;

    // Asignar memoria para pesos
    mlp->W1 = (float *)malloc(input_size * hidden_size * sizeof(float));
    mlp->b1 = (float *)calloc(hidden_size, sizeof(float)); // Inicializar a cero
    mlp->W2 = (float *)malloc(hidden_size * output_size * sizeof(float));
    mlp->b2 = (float *)calloc(output_size, sizeof(float));

    // Asignar memoria para activaciones
    mlp->Z1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    mlp->A1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    mlp->Z2 = (float *)malloc(batch_size * output_size * sizeof(float));
    mlp->A2 = (float *)malloc(batch_size * output_size * sizeof(float));

    // Asignar memoria para gradientes
    mlp->dW1 = (float *)malloc(input_size * hidden_size * sizeof(float));
    mlp->db1 = (float *)malloc(hidden_size * sizeof(float));
    mlp->dW2 = (float *)malloc(hidden_size * output_size * sizeof(float));
    mlp->db2 = (float *)malloc(output_size * sizeof(float));

    mlp->dZ2 = (float *)malloc(batch_size * output_size * sizeof(float));
    mlp->dA1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    mlp->dZ1 = (float *)malloc(batch_size * hidden_size * sizeof(float));

    // Inicialización de pesos con Xavier/He initialization
    // Ayuda a evitar vanishing/exploding gradients
    srand(seed);

    // W1: capa de entrada a oculta
    float limit1 = sqrtf(2.0f / (input_size + hidden_size));
    for (size_t i = 0; i < input_size * hidden_size; i++)
    {
        mlp->W1[i] = ((float)rand() / RAND_MAX) * 2.0f * limit1 - limit1;
    }

    // W2: capa oculta a salida
    float limit2 = sqrtf(2.0f / (hidden_size + output_size));
    for (size_t i = 0; i < hidden_size * output_size; i++)
    {
        mlp->W2[i] = ((float)rand() / RAND_MAX) * 2.0f * limit2 - limit2;
    }

    printf("MLP creado: %zu -> %zu -> %zu (batch_size=%zu)\n",
           input_size, hidden_size, output_size, batch_size);

    return mlp;
}

void mlp_free(MLP *mlp)
{
    if (!mlp)
        return;

    free(mlp->W1);
    free(mlp->b1);
    free(mlp->W2);
    free(mlp->b2);

    free(mlp->Z1);
    free(mlp->A1);
    free(mlp->Z2);
    free(mlp->A2);

    free(mlp->dW1);
    free(mlp->db1);
    free(mlp->dW2);
    free(mlp->db2);

    free(mlp->dZ2);
    free(mlp->dA1);
    free(mlp->dZ1);

    free(mlp);
}

void mlp_forward(MLP *mlp, const float *X, size_t actual_batch_size)
{
    // 1. Z1 = X @ W1 + b1
    matrix_multiply(X, mlp->W1, mlp->Z1, actual_batch_size, mlp->input_size, mlp->hidden_size);
    for (size_t b = 0; b < actual_batch_size; b++)
    {
        for (size_t j = 0; j < mlp->hidden_size; j++)
        {
            mlp->Z1[b * mlp->hidden_size + j] += mlp->b1[j];
        }
    }

    // 2. A1 = ReLU(Z1)
    relu(mlp->Z1, mlp->A1, actual_batch_size * mlp->hidden_size);

    // 3. Z2 = A1 @ W2 + b2
    matrix_multiply(mlp->A1, mlp->W2, mlp->Z2, actual_batch_size, mlp->hidden_size, mlp->output_size);
    for (size_t b = 0; b < actual_batch_size; b++)
    {
        for (size_t j = 0; j < mlp->output_size; j++)
        {
            mlp->Z2[b * mlp->output_size + j] += mlp->b2[j];
        }
    }

    // 4. A2 = Softmax(Z2)
    softmax(mlp->Z2, mlp->A2, actual_batch_size, mlp->output_size);
}

void mlp_backward(MLP *mlp, const float *X, const float *Y, size_t actual_batch_size)
{
    // 1. dZ2 = A2 - Y
    for (size_t i = 0; i < actual_batch_size * mlp->output_size; i++)
    {
        mlp->dZ2[i] = mlp->A2[i] - Y[i];
    }

    // 2. dW2 = A1^T @ dZ2
    matrix_transpose_multiply(mlp->A1, mlp->dZ2, mlp->dW2,
                              mlp->hidden_size, actual_batch_size, mlp->output_size);

    // 3. db2 = sum(dZ2, axis=0)
    for (size_t j = 0; j < mlp->output_size; j++)
    {
        mlp->db2[j] = 0.0f;
        for (size_t b = 0; b < actual_batch_size; b++)
        {
            mlp->db2[j] += mlp->dZ2[b * mlp->output_size + j];
        }
    }

    // 4. dA1 = dZ2 @ W2^T
    float *W2_T = (float *)malloc(mlp->output_size * mlp->hidden_size * sizeof(float));
    matrix_transpose(mlp->W2, W2_T, mlp->hidden_size, mlp->output_size);
    matrix_multiply(mlp->dZ2, W2_T, mlp->dA1, actual_batch_size, mlp->output_size, mlp->hidden_size);
    free(W2_T);

    // 5. dZ1 = dA1 * ReLU'(Z1)
    relu_derivative(mlp->Z1, mlp->dZ1, actual_batch_size * mlp->hidden_size);
    for (size_t i = 0; i < actual_batch_size * mlp->hidden_size; i++)
    {
        mlp->dZ1[i] *= mlp->dA1[i];
    }

    // 6. dW1 = X^T @ dZ1
    matrix_transpose_multiply(X, mlp->dZ1, mlp->dW1,
                              mlp->input_size, actual_batch_size, mlp->hidden_size);

    // 7. db1 = sum(dZ1, axis=0)
    for (size_t j = 0; j < mlp->hidden_size; j++)
    {
        mlp->db1[j] = 0.0f;
        for (size_t b = 0; b < actual_batch_size; b++)
        {
            mlp->db1[j] += mlp->dZ1[b * mlp->hidden_size + j];
        }
    }
}

void mlp_update_params(MLP *mlp, float learning_rate, size_t actual_batch_size)
{
    float lr_scaled = learning_rate / actual_batch_size;

    // Actualizar W1
    for (size_t i = 0; i < mlp->input_size * mlp->hidden_size; i++)
    {
        mlp->W1[i] -= lr_scaled * mlp->dW1[i];
    }

    // Actualizar b1
    for (size_t i = 0; i < mlp->hidden_size; i++)
    {
        mlp->b1[i] -= lr_scaled * mlp->db1[i];
    }

    // Actualizar W2
    for (size_t i = 0; i < mlp->hidden_size * mlp->output_size; i++)
    {
        mlp->W2[i] -= lr_scaled * mlp->dW2[i];
    }

    // Actualizar b2
    for (size_t i = 0; i < mlp->output_size; i++)
    {
        mlp->b2[i] -= lr_scaled * mlp->db2[i];
    }
}

float mlp_compute_loss(const MLP *mlp, const float *Y, size_t actual_batch_size)
{
    // TODO: Implementar Cross-Entropy Loss
    //
    // L = -sum(Y * log(A2)) / batch_size

    float loss = 0.0f;

    for (size_t b = 0; b < actual_batch_size; b++)
    {
        for (size_t i = 0; i < mlp->output_size; i++)
        {
            size_t idx = b * mlp->output_size + i;
            if (Y[idx] > 0.0f)
            {                                                 // Solo donde Y == 1
                loss -= Y[idx] * logf(mlp->A2[idx] + 1e-10f); // +epsilon para estabilidad
            }
        }
    }

    return loss / actual_batch_size;
}

float mlp_compute_accuracy(const MLP *mlp, const float *Y, size_t actual_batch_size)
{
    // TODO: Implementar cálculo de accuracy
    //
    // Para cada muestra:
    //   - Encontrar índice con max en A2 (predicción)
    //   - Encontrar índice con 1.0 en Y (label real)
    //   - Si coinciden, +1 acierto

    size_t correct = 0;

    for (size_t b = 0; b < actual_batch_size; b++)
    {
        // Encontrar clase predicha (argmax de A2)
        size_t pred_class = 0;
        float max_pred = mlp->A2[b * mlp->output_size];
        for (size_t i = 1; i < mlp->output_size; i++)
        {
            if (mlp->A2[b * mlp->output_size + i] > max_pred)
            {
                max_pred = mlp->A2[b * mlp->output_size + i];
                pred_class = i;
            }
        }

        // Encontrar clase real (argmax de Y)
        size_t true_class = 0;
        for (size_t i = 0; i < mlp->output_size; i++)
        {
            if (Y[b * mlp->output_size + i] > 0.5f)
            {
                true_class = i;
                break;
            }
        }

        if (pred_class == true_class)
        {
            correct++;
        }
    }

    return (float)correct / actual_batch_size;
}

void mlp_print_info(const MLP *mlp)
{
    printf("\n=== MLP Info ===\n");
    printf("Architecture: %zu -> %zu -> %zu\n",
           mlp->input_size, mlp->hidden_size, mlp->output_size);
    printf("Batch size: %zu\n", mlp->batch_size);
    printf("Parameters: %zu\n",
           mlp->input_size * mlp->hidden_size + mlp->hidden_size +
               mlp->hidden_size * mlp->output_size + mlp->output_size);
}
