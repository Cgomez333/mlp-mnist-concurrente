#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Dataset MNIST
Muestra imágenes del dataset para que veas qué está aprendiendo el modelo
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def leer_imagenes_mnist(archivo):
    """Lee archivo de imágenes MNIST (formato idx3-ubyte)"""
    with open(archivo, 'rb') as f:
        # Leer header
        magic, num_imagenes, filas, columnas = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f'Magic number incorrecto: {magic} (esperado: 2051)')
        
        # Leer todas las imágenes
        imagenes = np.frombuffer(f.read(), dtype=np.uint8)
        imagenes = imagenes.reshape(num_imagenes, filas, columnas)
        
        print(f"✅ Cargadas {num_imagenes} imágenes de {filas}×{columnas}")
        return imagenes

def leer_etiquetas_mnist(archivo):
    """Lee archivo de etiquetas MNIST (formato idx1-ubyte)"""
    with open(archivo, 'rb') as f:
        # Leer header
        magic, num_etiquetas = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f'Magic number incorrecto: {magic} (esperado: 2049)')
        
        # Leer todas las etiquetas
        etiquetas = np.frombuffer(f.read(), dtype=np.uint8)
        
        print(f"✅ Cargadas {num_etiquetas} etiquetas")
        return etiquetas

def visualizar_muestras(imagenes, etiquetas, cantidad=25, aleatorio=True):
    """Muestra una cuadrícula de imágenes del dataset"""
    
    # Calcular cuadrícula
    filas = int(np.sqrt(cantidad))
    columnas = int(np.ceil(cantidad / filas))
    
    # Seleccionar imágenes
    if aleatorio:
        indices = np.random.choice(len(imagenes), cantidad, replace=False)
    else:
        indices = range(cantidad)
    
    # Crear figura
    fig, axes = plt.subplots(filas, columnas, figsize=(15, 15))
    fig.suptitle('Ejemplos del Dataset MNIST', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < cantidad:
            idx = indices[i]
            ax.imshow(imagenes[idx], cmap='gray')
            ax.set_title(f'Dígito: {etiquetas[idx]}', fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def mostrar_por_digito(imagenes, etiquetas, digito, cantidad=10):
    """Muestra ejemplos de un dígito específico"""
    
    # Encontrar índices del dígito
    indices = np.where(etiquetas == digito)[0]
    
    print(f"\n📊 Hay {len(indices)} ejemplos del dígito '{digito}' en el dataset")
    
    # Seleccionar aleatoriamente
    indices_seleccionados = np.random.choice(indices, min(cantidad, len(indices)), replace=False)
    
    # Crear figura
    filas = 2
    columnas = 5
    fig, axes = plt.subplots(filas, columnas, figsize=(15, 6))
    fig.suptitle(f'10 ejemplos del dígito "{digito}" en MNIST', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(indices_seleccionados):
            idx = indices_seleccionados[i]
            ax.imshow(imagenes[idx], cmap='gray')
            ax.set_title(f'Índice: {idx}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def estadisticas_dataset(etiquetas):
    """Muestra estadísticas del dataset"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL DATASET MNIST")
    print("="*60)
    print(f"Total de imágenes: {len(etiquetas):,}")
    print(f"\nDistribución por dígito:")
    print("-" * 40)
    
    for digito in range(10):
        cantidad = np.sum(etiquetas == digito)
        porcentaje = (cantidad / len(etiquetas)) * 100
        barra = '█' * int(porcentaje)
        print(f"  Dígito {digito}: {cantidad:5,} imágenes ({porcentaje:5.2f}%) {barra}")
    
    print("="*60)

def main():
    """Función principal"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       🔍 Visualizador de Dataset MNIST                    ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # Rutas de archivos
    base_dir = Path(__file__).parent / 'backend' / 'data' / 'mnist'
    archivo_imagenes = base_dir / 'train-images-idx3-ubyte'
    archivo_etiquetas = base_dir / 'train-labels-idx1-ubyte'
    
    # Verificar que existen
    if not archivo_imagenes.exists():
        print(f"❌ No se encuentra: {archivo_imagenes}")
        print("   Asegúrate de estar en la raíz del proyecto")
        return
    
    print(f"📂 Cargando dataset desde: {base_dir}\n")
    
    # Cargar datos
    imagenes = leer_imagenes_mnist(archivo_imagenes)
    etiquetas = leer_etiquetas_mnist(archivo_etiquetas)
    
    # Mostrar estadísticas
    estadisticas_dataset(etiquetas)
    
    # Menú interactivo
    while True:
        print("\n" + "="*60)
        print("¿Qué quieres ver?")
        print("="*60)
        print("1. Ver 25 imágenes aleatorias")
        print("2. Ver ejemplos de un dígito específico (0-9)")
        print("3. Ver las primeras 25 imágenes")
        print("4. Ver estadísticas")
        print("5. Salir")
        print("="*60)
        
        opcion = input("\nElige una opción (1-5): ").strip()
        
        if opcion == '1':
            print("\n📸 Mostrando 25 imágenes aleatorias...\n")
            visualizar_muestras(imagenes, etiquetas, cantidad=25, aleatorio=True)
        
        elif opcion == '2':
            digito = input("¿Qué dígito quieres ver? (0-9): ").strip()
            if digito.isdigit() and 0 <= int(digito) <= 9:
                mostrar_por_digito(imagenes, etiquetas, int(digito), cantidad=10)
            else:
                print("❌ Dígito inválido. Debe ser entre 0 y 9.")
        
        elif opcion == '3':
            print("\n📸 Mostrando primeras 25 imágenes...\n")
            visualizar_muestras(imagenes, etiquetas, cantidad=25, aleatorio=False)
        
        elif opcion == '4':
            estadisticas_dataset(etiquetas)
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida. Elige 1-5.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrumpido por el usuario. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
