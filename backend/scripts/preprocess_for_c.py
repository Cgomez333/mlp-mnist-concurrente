#!/usr/bin/env python3
"""
Script para preprocesar MNIST y generar archivos binarios (.bin) para C.

Este script DEBE ser ejecutado por el compañero (Python) ANTES de que
el otro miembro (C) pueda trabajar.

Genera:
- train_images.bin: (60000, 784) float32
- train_labels.bin: (60000, 10) float32 one-hot
- test_images.bin: (10000, 784) float32
- test_labels.bin: (10000, 10) float32 one-hot

Formato binario: contiguous float32 array (little-endian)
"""

import os
import struct
import numpy as np

def read_mnist_images(filepath):
    """Lee archivo idx3-ubyte de imágenes MNIST"""
    with open(filepath, 'rb') as f:
        # Leer header
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2051, f"Magic number incorrecto: {magic}"
        
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        # Leer imágenes
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
    
    return images

def read_mnist_labels(filepath):
    """Lee archivo idx1-ubyte de etiquetas MNIST"""
    with open(filepath, 'rb') as f:
        # Leer header
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2049, f"Magic number incorrecto: {magic}"
        
        n_labels = struct.unpack('>I', f.read(4))[0]
        
        # Leer etiquetas
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

def preprocess_images(images):
    """
    Preprocesa imágenes:
    1. Aplanar 28x28 -> 784
    2. Normalizar [0, 255] -> [0, 1]
    3. Convertir a float32
    """
    n_images = images.shape[0]
    # Aplanar y normalizar
    images_flat = images.reshape(n_images, 784).astype(np.float32) / 255.0
    return images_flat

def labels_to_onehot(labels, num_classes=10):
    """Convierte etiquetas a one-hot encoding"""
    n_labels = labels.shape[0]
    onehot = np.zeros((n_labels, num_classes), dtype=np.float32)
    onehot[np.arange(n_labels), labels] = 1.0
    return onehot

def save_binary(data, filepath):
    """Guarda array NumPy como archivo binario float32"""
    data = data.astype(np.float32)
    with open(filepath, 'wb') as f:
        f.write(data.tobytes())
    print(f"✓ Guardado: {filepath} - Shape: {data.shape}, Size: {os.path.getsize(filepath) / (1024**2):.2f} MB")

def main():
    # Directorios
    mnist_dir = os.path.join('..', 'data', 'mnist')
    
    print("=" * 70)
    print("PREPROCESAMIENTO DE MNIST PARA C")
    print("=" * 70 + "\n")
    
    # Verificar que existan archivos raw
    required_files = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]
    
    for filename in required_files:
        filepath = os.path.join(mnist_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ Error: No se encuentra {filename}")
            print(f"   Ejecuta primero: python download_mnist.py")
            return
    
    print("📥 Cargando archivos MNIST raw...\n")
    
    # Cargar training set
    train_images_raw = read_mnist_images(os.path.join(mnist_dir, 'train-images-idx3-ubyte'))
    train_labels_raw = read_mnist_labels(os.path.join(mnist_dir, 'train-labels-idx1-ubyte'))
    
    print(f"Training set: {train_images_raw.shape[0]} imágenes")
    
    # Cargar test set
    test_images_raw = read_mnist_images(os.path.join(mnist_dir, 't10k-images-idx3-ubyte'))
    test_labels_raw = read_mnist_labels(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte'))
    
    print(f"Test set: {test_images_raw.shape[0]} imágenes\n")
    
    # Preprocesar
    print("🔧 Preprocesando...\n")
    
    train_images = preprocess_images(train_images_raw)
    train_labels = labels_to_onehot(train_labels_raw)
    
    test_images = preprocess_images(test_images_raw)
    test_labels = labels_to_onehot(test_labels_raw)
    
    # Guardar en formato binario
    print("💾 Guardando archivos .bin para C...\n")
    
    save_binary(train_images, os.path.join(mnist_dir, 'train_images.bin'))
    save_binary(train_labels, os.path.join(mnist_dir, 'train_labels.bin'))
    save_binary(test_images, os.path.join(mnist_dir, 'test_images.bin'))
    save_binary(test_labels, os.path.join(mnist_dir, 'test_labels.bin'))
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESAMIENTO COMPLETO")
    print("=" * 70)
    print("\nArchivos generados para C:")
    print("  - train_images.bin: (60000, 784) float32")
    print("  - train_labels.bin: (60000, 10) float32 one-hot")
    print("  - test_images.bin: (10000, 784) float32")
    print("  - test_labels.bin: (10000, 10) float32 one-hot")
    print("\n📝 Formato: Contiguous float32 arrays (little-endian)")
    print("📂 Ubicación:", os.path.abspath(mnist_dir))
    print("\n🚀 Ahora el compañero de C puede usar estos archivos con fread()")

if __name__ == '__main__':
    main()
