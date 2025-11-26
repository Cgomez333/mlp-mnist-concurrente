#!/usr/bin/env python3
"""
Script para descargar el dataset MNIST automáticamente.
Crea los archivos en data/mnist/
"""

import os
import urllib.request
import gzip
import shutil

# URLs del dataset MNIST original
MNIST_URLS = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

def download_and_extract(url, output_path):
    """Descarga y descomprime un archivo .gz"""
    print(f"Descargando {url}...")
    
    # Descargar archivo .gz
    gz_path = output_path + '.gz'
    urllib.request.urlretrieve(url, gz_path)
    
    # Descomprimir
    print(f"Descomprimiendo a {output_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Eliminar .gz
    os.remove(gz_path)
    print(f"✓ Listo: {output_path}\n")

def main():
    # Crear directorio si no existe
    data_dir = os.path.join('..', 'data', 'mnist')
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("Descargando dataset MNIST")
    print("=" * 60 + "\n")
    
    # Descargar cada archivo
    for name, url in MNIST_URLS.items():
        filename = os.path.basename(url).replace('.gz', '')
        output_path = os.path.join(data_dir, filename)
        
        # Si ya existe, preguntar
        if os.path.exists(output_path):
            print(f"⚠️  {filename} ya existe, omitiendo...")
            continue
        
        download_and_extract(url, output_path)
    
    print("=" * 60)
    print("✅ Descarga completa!")
    print(f"📂 Archivos en: {os.path.abspath(data_dir)}")
    print("=" * 60)
    
    # Mostrar tamaños
    print("\nArchivos descargados:")
    for name in MNIST_URLS.keys():
        filename = os.path.basename(MNIST_URLS[name]).replace('.gz', '')
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {filename}: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
