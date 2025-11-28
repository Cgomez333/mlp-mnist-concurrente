#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/**
 * Multiplicación de matrices: C = A * B
 * A: (rows_a x cols_a)
 * B: (cols_a x cols_b)
 * C: (rows_a x cols_b)
 */
void matrix_multiply(const float *A, const float *B, float *C,
                     size_t rows_a, size_t cols_a, size_t cols_b);

/**
 * Multiplicación de matriz por vector: y = A * x
 */
void matrix_vector_multiply(const float *A, const float *x, float *y,
                            size_t rows, size_t cols);

/**
 * Transpuesta de matriz: B = A^T
 */
void matrix_transpose(const float *A, float *B, size_t rows, size_t cols);

/**
 * Multiplicación con transpuesta: C = A^T * B
 * A: (rows_a x cols_a) -> transpuesta: (cols_a x rows_a)
 * B: (rows_a x cols_b)
 * C: (cols_a x cols_b)
 */
void matrix_transpose_multiply(const float *A, const float *B, float *C,
                               size_t cols_a, size_t rows_a, size_t cols_b);

/**
 * Suma elemento a elemento: C = A + B
 */
void matrix_add(const float *A, const float *B, float *C, size_t size);

/**
 * Resta elemento a elemento: C = A - B
 */
void matrix_subtract(const float *A, const float *B, float *C, size_t size);

/**
 * Multiplicación elemento a elemento (Hadamard product): C = A ⊙ B
 */
void matrix_elementwise_multiply(const float *A, const float *B, float *C, size_t size);

/**
 * Multiplicación por escalar: B = alpha * A
 */
void matrix_scale(const float *A, float alpha, float *B, size_t size);

/**
 * Función ReLU: y = max(0, x)
 */
void relu(const float *input, float *output, size_t size);

/**
 * Derivada de ReLU: y' = 1 if x > 0 else 0
 */
void relu_derivative(const float *input, float *output, size_t size);

/**
 * Función Softmax: y_i = exp(x_i) / sum(exp(x_j))
 * Aplicada por fila (batch)
 */
void softmax(const float *input, float *output, size_t batch_size, size_t num_classes);

/**
 * Suma por columnas (para calcular gradientes de bias)
 * input: (rows x cols)
 * output: (cols) - suma de cada columna
 */
void sum_columns(const float *input, float *output, size_t rows, size_t cols);

/**
 * Imprime una matriz (para debug)
 */
void print_matrix(const char *name, const float *matrix, size_t rows, size_t cols);

#endif // MATRIX_H
