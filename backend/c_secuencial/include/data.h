#ifndef DATA_H
#define DATA_H

#include <stddef.h>

/**
 * Estructura para almacenar el dataset MNIST
 */
typedef struct {
    float *images;      // Array de imágenes: [n_samples * 784]
    float *labels;      // Array de labels one-hot: [n_samples * 10]
    size_t n_samples;   // Número de muestras
    size_t img_size;    // Tamaño de cada imagen (784)
    size_t n_classes;   // Número de clases (10)
} Dataset;

/**
 * Carga el dataset desde archivos binarios .bin
 * 
 * @param images_path Ruta al archivo de imágenes (ej: "data/mnist/train_images.bin")
 * @param labels_path Ruta al archivo de labels (ej: "data/mnist/train_labels.bin")
 * @param n_samples Número de muestras a cargar
 * @return Dataset* Puntero a estructura con datos cargados, o NULL si error
 */
Dataset* load_dataset(const char *images_path, const char *labels_path, size_t n_samples);

/**
 * Libera memoria del dataset
 */
void free_dataset(Dataset *dataset);

/**
 * Imprime información del dataset
 */
void print_dataset_info(const Dataset *dataset);

/**
 * Obtiene un mini-batch del dataset
 * 
 * @param dataset Dataset completo
 * @param batch_images Array destino para imágenes del batch [batch_size * 784]
 * @param batch_labels Array destino para labels del batch [batch_size * 10]
 * @param batch_idx Índice del batch
 * @param batch_size Tamaño del batch
 */
void get_batch(const Dataset *dataset, float *batch_images, float *batch_labels,
               size_t batch_idx, size_t batch_size);

#endif // DATA_H
