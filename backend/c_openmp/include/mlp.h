#ifndef MLP_H
#define MLP_H

#include <stddef.h>

/**
 * Estructura que contiene todos los parámetros del MLP
 */
typedef struct {
    // Dimensiones
    size_t input_size;     // 784
    size_t hidden_size;    // 512
    size_t output_size;    // 10
    
    // Pesos y biases
    float *W1;  // (784 x 512)
    float *b1;  // (512)
    float *W2;  // (512 x 10)
    float *b2;  // (10)
    
    // Activaciones (para forward)
    float *Z1;  // (batch_size x 512) - pre-activación capa oculta
    float *A1;  // (batch_size x 512) - post-activación capa oculta (ReLU)
    float *Z2;  // (batch_size x 10) - pre-activación salida
    float *A2;  // (batch_size x 10) - post-activación salida (Softmax)
    
    // Gradientes (para backward)
    float *dW1; // (784 x 512)
    float *db1; // (512)
    float *dW2; // (512 x 10)
    float *db2; // (10)
    
    float *dZ2; // (batch_size x 10)
    float *dA1; // (batch_size x 512)
    float *dZ1; // (batch_size x 512)
    
    size_t batch_size;  // Tamaño del batch actual
} MLP;

/**
 * Inicializa el MLP con pesos aleatorios (Xavier initialization)
 * 
 * @param input_size Tamaño de entrada (784)
 * @param hidden_size Tamaño de capa oculta (512)
 * @param output_size Tamaño de salida (10)
 * @param batch_size Tamaño máximo de batch
 * @param seed Semilla para reproducibilidad
 * @return Puntero a MLP inicializado
 */
MLP* mlp_create(size_t input_size, size_t hidden_size, size_t output_size,
                size_t batch_size, unsigned int seed);

/**
 * Libera la memoria del MLP
 */
void mlp_free(MLP *mlp);

/**
 * Forward propagation
 * 
 * @param mlp Modelo MLP
 * @param X Entrada (batch_size x 784)
 * @param actual_batch_size Tamaño real del batch (puede ser menor que mlp->batch_size)
 */
void mlp_forward(MLP *mlp, const float *X, size_t actual_batch_size);

/**
 * Backward propagation
 * 
 * @param mlp Modelo MLP
 * @param X Entrada (batch_size x 784)
 * @param Y Labels one-hot (batch_size x 10)
 * @param actual_batch_size Tamaño real del batch
 */
void mlp_backward(MLP *mlp, const float *X, const float *Y, size_t actual_batch_size);

/**
 * Actualiza los pesos usando gradiente descendente
 * 
 * @param mlp Modelo MLP
 * @param learning_rate Tasa de aprendizaje
 * @param actual_batch_size Tamaño del batch (para promediar gradientes)
 */
void mlp_update_params(MLP *mlp, float learning_rate, size_t actual_batch_size);

/**
 * Calcula la pérdida Cross-Entropy
 * 
 * @param mlp Modelo MLP (usa mlp->A2 que contiene predicciones)
 * @param Y Labels one-hot (batch_size x 10)
 * @param actual_batch_size Tamaño del batch
 * @return Pérdida promedio
 */
float mlp_compute_loss(const MLP *mlp, const float *Y, size_t actual_batch_size);

/**
 * Calcula la precisión (accuracy)
 * 
 * @param mlp Modelo MLP (usa mlp->A2)
 * @param Y Labels one-hot (batch_size x 10)
 * @param actual_batch_size Tamaño del batch
 * @return Accuracy entre 0 y 1
 */
float mlp_compute_accuracy(const MLP *mlp, const float *Y, size_t actual_batch_size);

/**
 * Imprime información del modelo (para debug)
 */
void mlp_print_info(const MLP *mlp);

#endif // MLP_H
