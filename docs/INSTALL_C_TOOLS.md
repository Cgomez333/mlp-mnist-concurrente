# Instalación de Herramientas de Desarrollo C en Windows

## Opción 1: MinGW-w64 (Recomendada)

### Paso 1: Instalar MinGW

1. Descarga desde: https://sourceforge.net/projects/mingw-w64/
2. Ejecuta el instalador
3. Selecciona:
   - Architecture: x86_64
   - Threads: posix
   - Exception: seh

### Paso 2: Agregar a PATH

```powershell
# Agregar a variables de entorno
C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin
```

### Verificar instalación

```bash
gcc --version
make --version
```

## Opción 2: MSYS2 (Más completa)

### Instalar MSYS2

1. Descarga: https://www.msys2.org/
2. Ejecuta el instalador
3. Abre MSYS2 terminal y ejecuta:

```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc
pacman -S make
```

### Agregar a PATH

```
C:\msys64\mingw64\bin
```

## Opción 3: Visual Studio Build Tools

Si prefieres no usar Make, VSCode puede compilar directamente con tasks.

## Compilación Manual (si no tienes make)

### Windows PowerShell

```powershell
# Crear directorios
mkdir build, bin

# Compilar
gcc -Wall -Wextra -O3 -Iinclude -c src/data.c -o build/data.o
gcc -Wall -Wextra -O3 -Iinclude -c src/matrix.c -o build/matrix.o
gcc -Wall -Wextra -O3 -Iinclude -c src/mlp.c -o build/mlp.o
gcc -Wall -Wextra -O3 -Iinclude -c src/train.c -o build/train.o

# Enlazar
gcc build/data.o build/matrix.o build/mlp.o build/train.o -lm -o bin/train_seq.exe

# Ejecutar
.\bin\train_seq.exe
```

## Script de Compilación (compile.bat)

Guarda esto como `compile.bat` en `c_secuencial/`:

```bat
@echo off
echo Compilando MLP MNIST - C Secuencial...

if not exist build mkdir build
if not exist bin mkdir bin

echo Compilando data.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/data.c -o build/data.o
if errorlevel 1 goto error

echo Compilando matrix.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/matrix.c -o build/matrix.o
if errorlevel 1 goto error

echo Compilando mlp.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/mlp.c -o build/mlp.o
if errorlevel 1 goto error

echo Compilando train.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/train.c -o build/train.o
if errorlevel 1 goto error

echo Enlazando...
gcc build/data.o build/matrix.o build/mlp.o build/train.o -lm -o bin/train_seq.exe
if errorlevel 1 goto error

echo.
echo Compilacion exitosa: bin\train_seq.exe
echo.
echo Ejecutar con: .\bin\train_seq.exe
goto end

:error
echo.
echo ERROR: Compilacion fallida
exit /b 1

:end
```

Luego ejecuta:

```cmd
compile.bat
```

## Verificación

Después de instalar GCC:

```bash
cd c_secuencial
make
# o
./compile.bat
```

Deberías ver:

```
Compilando data.c...
Compilando matrix.c...
Compilando mlp.c...
Compilando train.c...
Enlazando...
Build complete: bin/train_seq
```
