# Makefile para MLP MNIST - Versión Secuencial en C

CC = gcc
CFLAGS = -Wall -Wextra -O3 -Iinclude
LDFLAGS = -lm

# Directorios
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Archivos fuente
SOURCES = $(SRC_DIR)/train.c $(SRC_DIR)/mlp.c $(SRC_DIR)/matrix.c $(SRC_DIR)/data.c
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Ejecutable
TARGET = $(BIN_DIR)/train_seq

# Regla principal
all: directories $(TARGET)

# Crear directorios si no existen
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Compilar ejecutable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# Compilar archivos objeto
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Ejecutar
run: $(TARGET)
	@echo "Running C Sequential MLP..."
	cd $(BIN_DIR) && ./train_seq

# Limpiar
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Recompilar todo
rebuild: clean all

# Ayuda
help:
	@echo "Makefile para MLP MNIST - C Secuencial"
	@echo ""
	@echo "Uso:"
	@echo "  make          - Compilar el proyecto"
	@echo "  make run      - Compilar y ejecutar"
	@echo "  make clean    - Eliminar archivos compilados"
	@echo "  make rebuild  - Limpiar y recompilar"
	@echo "  make help     - Mostrar esta ayuda"

.PHONY: all directories run clean rebuild help
