/*
 * Exportador de pesos del modelo MLP a formato JSON
 * Para usar predicciones reales en el frontend
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/mlp.h"
#include "../include/data.h"

void export_weights_to_json(MLP *mlp, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: No se pudo crear el archivo %s\n", filename);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"input_size\": %d,\n", mlp->input_size);
    fprintf(fp, "  \"hidden_size\": %d,\n", mlp->hidden_size);
    fprintf(fp, "  \"output_size\": %d,\n", mlp->output_size);

    // Exportar W1
    fprintf(fp, "  \"W1\": [");
    for (int i = 0; i < mlp->input_size * mlp->hidden_size; i++)
    {
        fprintf(fp, "%.6f", mlp->W1[i]);
        if (i < mlp->input_size * mlp->hidden_size - 1)
            fprintf(fp, ",");
    }
    fprintf(fp, "],\n");

    // Exportar b1
    fprintf(fp, "  \"b1\": [");
    for (int i = 0; i < mlp->hidden_size; i++)
    {
        fprintf(fp, "%.6f", mlp->b1[i]);
        if (i < mlp->hidden_size - 1)
            fprintf(fp, ",");
    }
    fprintf(fp, "],\n");

    // Exportar W2
    fprintf(fp, "  \"W2\": [");
    for (int i = 0; i < mlp->hidden_size * mlp->output_size; i++)
    {
        fprintf(fp, "%.6f", mlp->W2[i]);
        if (i < mlp->hidden_size * mlp->output_size - 1)
            fprintf(fp, ",");
    }
    fprintf(fp, "],\n");

    // Exportar b2
    fprintf(fp, "  \"b2\": [");
    for (int i = 0; i < mlp->output_size; i++)
    {
        fprintf(fp, "%.6f", mlp->b2[i]);
        if (i < mlp->output_size - 1)
            fprintf(fp, ",");
    }
    fprintf(fp, "]\n");

    fprintf(fp, "}\n");

    fclose(fp);
    printf("✓ Pesos exportados a: %s\n", filename);
}

int main()
{
    printf("=================================================================\n");
    printf("Exportador de Pesos MLP - Para Frontend\n");
    printf("=================================================================\n\n");

    // Cargar dataset para entrenar
    printf("Cargando dataset...\n");
    Dataset *train = load_dataset("../data/mnist/train_images.bin",
                                  "../data/mnist/train_labels.bin",
                                  60000);
    if (!train)
    {
        fprintf(stderr, "Error al cargar dataset\n");
        return 1;
    }

    // Crear y entrenar modelo
    printf("Creando modelo MLP...\n");
    int batch_size = 64;
    MLP *mlp = mlp_create(784, 512, 10, batch_size, 42);

    printf("Entrenando modelo (esto tomará tiempo)...\n");
    printf("Solo 1 época para exportación rápida...\n\n");

    // Entrenar 1 época
    for (int epoch = 0; epoch < 10; epoch++)
    {
        size_t num_batches = train->n_samples / batch_size;

        float *batch_images = (float *)malloc(batch_size * 784 * sizeof(float));
        float *batch_labels = (float *)malloc(batch_size * 10 * sizeof(float));

        if (!batch_images || !batch_labels)
        {
            fprintf(stderr, "Error al alocar memoria para batches\n");
            mlp_free(mlp);
            free_dataset(train);
            return 1;
        }

        for (size_t batch = 0; batch < num_batches; batch++)
        {
            // get_batch ya multiplica batch por batch_size internamente
            get_batch(train, batch_images, batch_labels, batch, batch_size);

            mlp_forward(mlp, batch_images, batch_size);
            mlp_backward(mlp, batch_images, batch_labels, batch_size);
            mlp_update_params(mlp, 0.01, batch_size);

            if (batch % 100 == 0)
            {
                printf("\rBatch %zu/%zu", batch, num_batches);
                fflush(stdout);
            }
        }

        free(batch_images);
        free(batch_labels);
    }

    printf("\n\n✓ Entrenamiento completado\n");

    // Exportar pesos
    printf("Exportando pesos a JSON...\n");
    export_weights_to_json(mlp, "../api/model_weights.json");

    // Cleanup
    mlp_free(mlp);
    free_dataset(train);

    printf("\n=================================================================\n");
    printf("✓ Proceso completado exitosamente\n");
    printf("Los pesos están en: backend/api/model_weights.json\n");
    printf("=================================================================\n");

    return 0;
}
