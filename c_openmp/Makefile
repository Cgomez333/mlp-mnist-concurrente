# Makefile para MLP-MNIST con OpenMP
# Compilación paralela usando OpenMP

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp
INCLUDES = -Iinclude
LIBS = -lm

# Directorios
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Archivos fuente y objetos
SOURCES = $(SRC_DIR)/data.c $(SRC_DIR)/matrix.c $(SRC_DIR)/mlp.c $(SRC_DIR)/train.c
OBJECTS = $(BUILD_DIR)/data.o $(BUILD_DIR)/matrix.o $(BUILD_DIR)/mlp.o $(BUILD_DIR)/train.o

# Ejecutable
TARGET = $(BIN_DIR)/train_openmp.exe

# Regla principal
all: $(TARGET)

# Compilar ejecutable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $(TARGET)
	@echo "✓ Compilación exitosa: $(TARGET)"

# Compilar objetos
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Crear directorios si no existen
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Limpiar archivos compilados
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)
	@echo "✓ Archivos compilados eliminados"

# Ejecutar entrenamiento con número de threads especificado
# Uso: make run THREADS=4
run: $(TARGET)
	@if [ -z "$(THREADS)" ]; then \
		echo "Ejecutando con número de threads por defecto..."; \
		$(TARGET); \
	else \
		echo "Ejecutando con $(THREADS) threads..."; \
		OMP_NUM_THREADS=$(THREADS) $(TARGET); \
	fi

# Benchmarks con diferentes números de threads
benchmark: $(TARGET)
	@echo "=== Benchmark OpenMP ==="
	@echo "1 thread:"
	@OMP_NUM_THREADS=1 $(TARGET)
	@echo "\n2 threads:"
	@OMP_NUM_THREADS=2 $(TARGET)
	@echo "\n4 threads:"
	@OMP_NUM_THREADS=4 $(TARGET)
	@echo "\n8 threads:"
	@OMP_NUM_THREADS=8 $(TARGET)

.PHONY: all clean run benchmark
