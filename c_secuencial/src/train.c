/**
 * train.c - Programa principal de entrenamiento MLP (versión secuencial)
 *
 * Entrena una red neuronal para clasificación de dígitos MNIST
 * Exporta métricas (loss, accuracy, tiempo) a CSV para análisis posterior
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data.h"
#include "mlp.h"

// Hiperparámetros definidos según experiment_design.md
#define EPOCHS 10
#define BATCH_SIZE 64
#define LEARNING_RATE 0.01f
#define HIDDEN_NEURONS 512
#define RANDOM_SEED 42

int main()
{
    printf("=================================================================\n");
    printf("MLP MNIST - Implementación Secuencial en C\n");
    printf("=================================================================\n\n");

    // Cargar dataset
    printf("Cargando dataset de entrenamiento...\n");
    Dataset *train = load_dataset("../../data/mnist/train_images.bin",
                                  "../../data/mnist/train_labels.bin",
                                  60000);
    if (!train)
    {
        fprintf(stderr, "Error al cargar dataset de entrenamiento\n");
        return 1;
    }

    printf("Cargando dataset de prueba...\n");
    Dataset *test = load_dataset("../../data/mnist/test_images.bin",
                                 "../../data/mnist/test_labels.bin",
                                 10000);
    if (!test)
    {
        fprintf(stderr, "Error al cargar dataset de prueba\n");
        free_dataset(train);
        return 1;
    }

    print_dataset_info(train);

    // Crear modelo MLP
    printf("\nCreando modelo MLP...\n");
    MLP *mlp = mlp_create(784, HIDDEN_NEURONS, 10, BATCH_SIZE, RANDOM_SEED);
    if (!mlp)
    {
        fprintf(stderr, "Error al crear MLP\n");
        free_dataset(train);
        free_dataset(test);
        return 1;
    }

    mlp_print_info(mlp);

    // Preparar batches
    size_t num_batches = (train->n_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    float *batch_images = (float *)malloc(BATCH_SIZE * 784 * sizeof(float));
    float *batch_labels = (float *)malloc(BATCH_SIZE * 10 * sizeof(float));

    if (!batch_images || !batch_labels)
    {
        fprintf(stderr, "Error al asignar memoria para batches\n");
        mlp_free(mlp);
        free_dataset(train);
        free_dataset(test);
        return 1;
    }

    printf("\n=================================================================\n");
    printf("Iniciando entrenamiento (%d epochs, batch_size=%d, lr=%.3f)\n",
           EPOCHS, BATCH_SIZE, LEARNING_RATE);
    printf("=================================================================\n\n");

    // Medir tiempo total
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Entrenamiento
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        struct timespec epoch_start, epoch_end;
        clock_gettime(CLOCK_MONOTONIC, &epoch_start);

        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;

        // Iterar sobre batches
        for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++)
        {
            // Obtener batch
            get_batch(train, batch_images, batch_labels, batch_idx, BATCH_SIZE);

            size_t actual_batch_size = BATCH_SIZE;
            if ((batch_idx + 1) * BATCH_SIZE > train->n_samples)
            {
                actual_batch_size = train->n_samples - batch_idx * BATCH_SIZE;
            }

            // Forward pass
            mlp_forward(mlp, batch_images, actual_batch_size);

            // Calcular loss y accuracy
            float batch_loss = mlp_compute_loss(mlp, batch_labels, actual_batch_size);
            float batch_acc = mlp_compute_accuracy(mlp, batch_labels, actual_batch_size);

            epoch_loss += batch_loss;
            epoch_accuracy += batch_acc;

            // Backward pass
            mlp_backward(mlp, batch_images, batch_labels, actual_batch_size);

            // Actualizar parámetros
            mlp_update_params(mlp, LEARNING_RATE, actual_batch_size);
        }

        clock_gettime(CLOCK_MONOTONIC, &epoch_end);
        double epoch_time = (epoch_end.tv_sec - epoch_start.tv_sec) +
                            (epoch_end.tv_nsec - epoch_start.tv_nsec) / 1e9;

        // Promediar métricas
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;

        printf("Epoch %2d/%d - Loss: %.4f - Accuracy: %.4f - Time: %.2fs\n",
               epoch + 1, EPOCHS, epoch_loss, epoch_accuracy, epoch_time);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double total_time = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("\n=================================================================\n");
    printf("Entrenamiento completado en %.2f segundos\n", total_time);
    printf("=================================================================\n\n");

    // Evaluación en test set
    printf("Evaluando en test set...\n");
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    size_t test_batches = (test->n_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    for (size_t batch_idx = 0; batch_idx < test_batches; batch_idx++)
    {
        get_batch(test, batch_images, batch_labels, batch_idx, BATCH_SIZE);

        size_t actual_batch_size = BATCH_SIZE;
        if ((batch_idx + 1) * BATCH_SIZE > test->n_samples)
        {
            actual_batch_size = test->n_samples - batch_idx * BATCH_SIZE;
        }

        mlp_forward(mlp, batch_images, actual_batch_size);
        test_loss += mlp_compute_loss(mlp, batch_labels, actual_batch_size);
        test_accuracy += mlp_compute_accuracy(mlp, batch_labels, actual_batch_size);
    }

    test_loss /= test_batches;
    test_accuracy /= test_batches;

    printf("Test Loss: %.4f - Test Accuracy: %.4f\n\n", test_loss, test_accuracy);

    // Guardar resultados en CSV
    printf("Guardando resultados...\n");
    FILE *csv = fopen("../../results/raw/c_sequential.csv", "w");
    if (csv)
    {
        fprintf(csv, "implementation,language,parallelization,workers_threads,batch_size,epochs,learning_rate,hidden_neurons,total_time_sec,avg_epoch_time,final_loss,final_accuracy,speedup_vs_baseline,notes\n");
        fprintf(csv, "c_seq,c,none,1,%d,%d,%.3f,%d,%.2f,%.2f,%.4f,%.4f,1.00,baseline_c\n",
                BATCH_SIZE, EPOCHS, LEARNING_RATE, HIDDEN_NEURONS,
                total_time, total_time / EPOCHS, test_loss, test_accuracy);
        fclose(csv);
        printf("Resultados guardados en: results/raw/c_sequential.csv\n");
    }

    // Liberar memoria
    free(batch_images);
    free(batch_labels);
    mlp_free(mlp);
    free_dataset(train);
    free_dataset(test);

    printf("\n=================================================================\n");
    printf("Programa finalizado exitosamente\n");
    printf("=================================================================\n");

    return 0;
}
