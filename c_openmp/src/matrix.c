/**
 * matrix.c - Operaciones matriciales para MLP (versión OpenMP)
 *
 * Implementa multiplicación de matrices, funciones de activación (ReLU, Softmax)
 * y operaciones auxiliares necesarias para backpropagation
 * Paralelizado con OpenMP para mejor rendimiento
 */

#include "matrix.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

void matrix_multiply(const float *A, const float *B, float *C,
                     size_t rows_a, size_t cols_a, size_t cols_b)
{

    // A: (rows_a x cols_a)
    // B: (cols_a x cols_b)
    // C: (rows_a x cols_b)

    // Inicializar C a cero
    memset(C, 0, rows_a * cols_b * sizeof(float));

// Triple bucle paralelizado para multiplicación
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows_a; i++)
    {
        for (size_t j = 0; j < cols_b; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++)
            {
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}

void matrix_transpose(const float *A, float *B, size_t rows, size_t cols)
{

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_transpose_multiply(const float *A, const float *B, float *C,
                               size_t cols_a, size_t rows_a, size_t cols_b)
{
// C = A^T * B
// A: (rows_a x cols_a) -> A^T: (cols_a x rows_a)
// B: (rows_a x cols_b)
// C: (cols_a x cols_b)
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < cols_a; i++)
    {
        for (size_t j = 0; j < cols_b; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < rows_a; k++)
            {
                // A^T[i][k] = A[k][i]
                sum += A[k * cols_a + i] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}

void matrix_add(const float *A, const float *B, float *C, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void matrix_subtract(const float *A, const float *B, float *C, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        C[i] = A[i] - B[i];
    }
}

void matrix_elementwise_multiply(const float *A, const float *B, float *C, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        C[i] = A[i] * B[i];
    }
}

void matrix_scale(const float *A, float alpha, float *B, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        B[i] = alpha * A[i];
    }
}

void relu(const float *input, float *output, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

void relu_derivative(const float *input, float *output, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        output[i] = (input[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void softmax(const float *input, float *output, size_t batch_size, size_t num_classes)
{

// Paralelizar por cada muestra en el batch
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; b++)
    {
        const float *row_in = input + b * num_classes;
        float *row_out = output + b * num_classes;

        // Encontrar el máximo (para estabilidad numérica)
        float max_val = row_in[0];
        for (size_t i = 1; i < num_classes; i++)
        {
            if (row_in[i] > max_val)
            {
                max_val = row_in[i];
            }
        }

        // Calcular exp(x - max) y suma
        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; i++)
        {
            row_out[i] = expf(row_in[i] - max_val);
            sum += row_out[i];
        }

        // Normalizar
        for (size_t i = 0; i < num_classes; i++)
        {
            row_out[i] /= sum;
        }
    }
}

void sum_columns(const float *input, float *output, size_t rows, size_t cols)
{

    // Inicializar output a cero
    memset(output, 0, cols * sizeof(float));

    // Sumar cada fila
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            output[j] += input[i * cols + j];
        }
    }
}

void print_matrix(const char *name, const float *matrix, size_t rows, size_t cols)
{
    printf("\n%s (%zux%zu):\n", name, rows, cols);

    // Limitar a 5x5 para no saturar la consola
    size_t max_rows = (rows > 5) ? 5 : rows;
    size_t max_cols = (cols > 5) ? 5 : cols;

    for (size_t i = 0; i < max_rows; i++)
    {
        for (size_t j = 0; j < max_cols; j++)
        {
            printf("%.4f ", matrix[i * cols + j]);
        }
        if (cols > 5)
            printf("...");
        printf("\n");
    }
    if (rows > 5)
        printf("...\n");
}
