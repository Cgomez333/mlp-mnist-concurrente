/**
 * data.c - Carga y gestión del dataset MNIST
 *
 * Lee archivos binarios preprocesados (.bin) con imágenes y etiquetas
 * Proporciona funciones para obtener mini-batches durante entrenamiento
 */

#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Dataset *load_dataset(const char *images_path, const char *labels_path, size_t n_samples)
{
    // Asignar memoria para la estructura
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    if (!dataset)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para Dataset\n");
        return NULL;
    }

    dataset->n_samples = n_samples;
    dataset->img_size = 784;
    dataset->n_classes = 10;

    // Asignar memoria para imágenes y labels
    dataset->images = (float *)malloc(n_samples * 784 * sizeof(float));
    dataset->labels = (float *)malloc(n_samples * 10 * sizeof(float));

    if (!dataset->images || !dataset->labels)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para datos\n");
        free_dataset(dataset);
        return NULL;
    }

    // Cargar imágenes
    FILE *img_file = fopen(images_path, "rb");
    if (!img_file)
    {
        fprintf(stderr, "Error: No se pudo abrir %s\n", images_path);
        fprintf(stderr, "¿Tu compañero ya ejecutó preprocess_for_c.py?\n");
        free_dataset(dataset);
        return NULL;
    }

    size_t img_read = fread(dataset->images, sizeof(float), n_samples * 784, img_file);
    fclose(img_file);

    if (img_read != n_samples * 784)
    {
        fprintf(stderr, "Error: Se esperaban %zu elementos, se leyeron %zu\n",
                n_samples * 784, img_read);
        free_dataset(dataset);
        return NULL;
    }

    // Cargar labels
    FILE *lbl_file = fopen(labels_path, "rb");
    if (!lbl_file)
    {
        fprintf(stderr, "Error: No se pudo abrir %s\n", labels_path);
        free_dataset(dataset);
        return NULL;
    }

    size_t lbl_read = fread(dataset->labels, sizeof(float), n_samples * 10, lbl_file);
    fclose(lbl_file);

    if (lbl_read != n_samples * 10)
    {
        fprintf(stderr, "Error: Se esperaban %zu elementos, se leyeron %zu\n",
                n_samples * 10, lbl_read);
        free_dataset(dataset);
        return NULL;
    }

    printf("✓ Dataset cargado exitosamente\n");
    printf("  - %zu imágenes de %zu características\n", n_samples, dataset->img_size);
    printf("  - Labels one-hot de %zu clases\n", dataset->n_classes);

    return dataset;
}

void free_dataset(Dataset *dataset)
{
    if (dataset)
    {
        if (dataset->images)
            free(dataset->images);
        if (dataset->labels)
            free(dataset->labels);
        free(dataset);
    }
}

void print_dataset_info(const Dataset *dataset)
{
    if (!dataset)
        return;

    printf("\n=== Dataset Info ===\n");
    printf("Muestras: %zu\n", dataset->n_samples);
    printf("Imagen size: %zu\n", dataset->img_size);
    printf("Clases: %zu\n", dataset->n_classes);
    printf("Memoria imágenes: %.2f MB\n",
           (dataset->n_samples * dataset->img_size * sizeof(float)) / (1024.0 * 1024.0));
    printf("Memoria labels: %.2f MB\n",
           (dataset->n_samples * dataset->n_classes * sizeof(float)) / (1024.0 * 1024.0));

    // Mostrar primera imagen (primeros 10 valores)
    printf("\nPrimera imagen (primeros 10 valores): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%.3f ", dataset->images[i]);
    }
    printf("...\n");

    // Mostrar primera label
    printf("Primera label (one-hot): ");
    for (int i = 0; i < 10; i++)
    {
        printf("%.0f ", dataset->labels[i]);
    }
    printf("\n");
}

void get_batch(const Dataset *dataset, float *batch_images, float *batch_labels,
               size_t batch_idx, size_t batch_size)
{
    size_t start_idx = batch_idx * batch_size;

    // Verificar que no exceda el tamaño del dataset
    if (start_idx + batch_size > dataset->n_samples)
    {
        batch_size = dataset->n_samples - start_idx;
    }

    // Copiar imágenes del batch
    memcpy(batch_images,
           dataset->images + start_idx * dataset->img_size,
           batch_size * dataset->img_size * sizeof(float));

    // Copiar labels del batch
    memcpy(batch_labels,
           dataset->labels + start_idx * dataset->n_classes,
           batch_size * dataset->n_classes * sizeof(float));
}
