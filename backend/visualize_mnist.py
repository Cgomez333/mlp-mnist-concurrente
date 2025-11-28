#!/usr/bin/env python3
"""
Visualizador de imágenes MNIST
Permite ver las imágenes del dataset en formato IDX (original)
"""

import struct
import sys

def load_mnist_images_idx(filename):
    """Carga todas las imágenes del archivo IDX"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = []
        for _ in range(num):
            image = struct.unpack('B' * (rows * cols), f.read(rows * cols))
            # Normalizar a 0.0-1.0
            pixels = [pixel / 255.0 for pixel in image]
            images.append(pixels)
        return images

def load_mnist_labels_idx(filename):
    """Carga todas las etiquetas del archivo IDX"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = struct.unpack('B' * num, f.read(num))
        return labels

def load_mnist_image(images, labels, index=0):
    """Obtiene una imagen específica del dataset cargado"""
    return images[index], labels[index]


def print_image(pixels, label):
    """Imprime la imagen en la consola usando caracteres ASCII"""
    
    print(f"\n{'='*60}")
    print(f"  ETIQUETA: {label}")
    print(f"{'='*60}\n")
    
    # Convertir a matriz 28x28
    for row in range(28):
        line = ""
        for col in range(28):
            pixel_value = pixels[row * 28 + col]
            
            # Convertir a caracteres según intensidad
            if pixel_value < 0.1:
                char = '  '  # Blanco
            elif pixel_value < 0.3:
                char = '░░'  # Gris claro
            elif pixel_value < 0.5:
                char = '▒▒'  # Gris medio
            elif pixel_value < 0.7:
                char = '▓▓'  # Gris oscuro
            else:
                char = '██'  # Negro
            
            line += char
        print(line)
    
    print(f"\n{'='*60}\n")


def show_multiple_images(images, labels, start=0, count=5):
    """Muestra múltiples imágenes consecutivas"""
    
    for i in range(start, start + count):
        pixels, label = load_mnist_image(images, labels, i)
        print(f"\nImagen #{i}")
        print_image(pixels, label)
        
        if i < start + count - 1:
            input("Presiona ENTER para ver la siguiente imagen...")


def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python visualize_mnist.py <numero_imagen>")
        print("  python visualize_mnist.py <inicio> <cantidad>")
        print("\nEjemplos:")
        print("  python visualize_mnist.py 0           # Ver imagen 0")
        print("  python visualize_mnist.py 100         # Ver imagen 100")
        print("  python visualize_mnist.py 0 10        # Ver imágenes 0-9")
        print("\nDataset:")
        print("  - Training: 60,000 imágenes (índices 0-59999)")
        print("  - Test: 10,000 imágenes (índices 0-9999)")
        sys.exit(1)
    
    # Rutas a los archivos IDX originales
    train_images_file = "data/mnist/train-images-idx3-ubyte"
    train_labels_file = "data/mnist/train-labels-idx1-ubyte"
    test_images_file = "data/mnist/t10k-images-idx3-ubyte"
    test_labels_file = "data/mnist/t10k-labels-idx1-ubyte"
    
    # Pedir al usuario qué dataset usar
    print("\n¿Qué dataset quieres visualizar?")
    print("1. Training (60,000 imágenes)")
    print("2. Test (10,000 imágenes)")
    
    choice = input("\nElige (1/2) [default: 1]: ").strip()
    
    if choice == '2':
        image_file = test_images_file
        label_file = test_labels_file
        max_images = 10000
        dataset_name = "Test"
    else:
        image_file = train_images_file
        label_file = train_labels_file
        max_images = 60000
        dataset_name = "Training"
    
    print(f"\n✓ Dataset seleccionado: {dataset_name}")
    print(f"⏳ Cargando {max_images} imágenes...")
    
    # Cargar todo el dataset
    images = load_mnist_images_idx(image_file)
    labels = load_mnist_labels_idx(label_file)
    
    print(f"✓ Dataset cargado correctamente")
    
    # Procesar argumentos
    if len(sys.argv) == 2:
        # Ver una sola imagen
        index = int(sys.argv[1])
        if index >= max_images:
            print(f"Error: índice {index} fuera de rango (max: {max_images-1})")
            sys.exit(1)
        
        pixels, label = load_mnist_image(images, labels, index)
        print(f"\nImagen #{index} del dataset {dataset_name}")
        print_image(pixels, label)
        
    else:
        # Ver múltiples imágenes
        start = int(sys.argv[1])
        count = int(sys.argv[2])
        
        if start + count > max_images:
            print(f"Error: rango excede el dataset (max: {max_images})")
            sys.exit(1)
        
        show_multiple_images(images, labels, start, count)


if __name__ == "__main__":
    main()
