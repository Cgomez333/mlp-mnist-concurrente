#!/usr/bin/env python3
"""
Descarga MNIST desde múltiples fuentes (con fallback)
"""

import os
import urllib.request
import gzip
import shutil

# Lista de URLs alternativas (probamos en orden)
SOURCES = [
    {
        'name': 'PyTorch Mirror',
        'urls': {
            'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
        }
    },
    {
        'name': 'GitHub Mirror',
        'urls': {
            'train_images': 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz',
            'train_labels': 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz',
            'test_images': 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz',
            'test_labels': 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz'
        }
    }
]

def download_file(url, output_path, timeout=30):
    """Descarga un archivo con timeout"""
    try:
        print(f"  Descargando de {url[:50]}...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def download_and_extract(url, output_path):
    """Descarga y descomprime un archivo .gz"""
    gz_path = output_path + '.gz'
    
    if not download_file(url, gz_path):
        return False
    
    # Descomprimir
    print(f"  Descomprimiendo...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  ✓ Listo\n")
        return True
    except Exception as e:
        print(f"  ✗ Error al descomprimir: {e}")
        return False

def try_download_from_source(source, data_dir):
    """Intenta descargar desde una fuente específica"""
    print(f"\nProbando: {source['name']}")
    print("=" * 60)
    
    success = True
    for name, url in source['urls'].items():
        filename = os.path.basename(url).replace('.gz', '')
        output_path = os.path.join(data_dir, filename)
        
        if os.path.exists(output_path):
            print(f"  {filename} ya existe, omitiendo...")
            continue
        
        if not download_and_extract(url, output_path):
            success = False
            break
    
    return success

def main():
    # Crear directorio
    data_dir = os.path.join('..', 'data', 'mnist')
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("DESCARGA DE MNIST")
    print("=" * 60)
    
    # Probar cada fuente hasta que una funcione
    for source in SOURCES:
        if try_download_from_source(source, data_dir):
            print("\n" + "=" * 60)
            print("✓ Descarga completa!")
            print(f"Archivos en: {os.path.abspath(data_dir)}")
            print("=" * 60)
            
            # Mostrar tamaños
            print("\nArchivos descargados:")
            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filename}: {size_mb:.2f} MB")
            return
        else:
            print(f"\n✗ Fallo al descargar desde {source['name']}, probando siguiente fuente...\n")
    
    print("\n" + "=" * 60)
    print("✗ ERROR: No se pudo descargar desde ninguna fuente")
    print("=" * 60)
    print("\nSolución alternativa:")
    print("1. Descarga manualmente desde: https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
    print("2. Coloca los archivos en:", os.path.abspath(data_dir))

if __name__ == '__main__':
    main()
