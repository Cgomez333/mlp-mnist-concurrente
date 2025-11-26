@echo off
echo ============================================
echo  MLP MNIST - Compilacion C OpenMP
echo ============================================
echo.

REM Crear directorios
if not exist build mkdir build
if not exist bin mkdir bin
if not exist ..\results mkdir ..\results
if not exist ..\results\raw mkdir ..\results\raw

echo [1/5] Compilando data.c...
gcc -Wall -Wextra -O3 -fopenmp -Iinclude -c src/data.c -o build/data.o
if errorlevel 1 goto error

echo [2/5] Compilando matrix.c con OpenMP...
gcc -Wall -Wextra -O3 -fopenmp -Iinclude -c src/matrix.c -o build/matrix.o
if errorlevel 1 goto error

echo [3/5] Compilando mlp.c con OpenMP...
gcc -Wall -Wextra -O3 -fopenmp -Iinclude -c src/mlp.c -o build/mlp.o
if errorlevel 1 goto error

echo [4/5] Compilando train.c con OpenMP...
gcc -Wall -Wextra -O3 -fopenmp -Iinclude -c src/train.c -o build/train.o
if errorlevel 1 goto error

echo [5/5] Enlazando con OpenMP...
gcc -fopenmp build/data.o build/matrix.o build/mlp.o build/train.o -lm -o bin/train_omp.exe
if errorlevel 1 goto error

echo.
echo ============================================
echo  COMPILACION EXITOSA
echo ============================================
echo.
echo Ejecutable: bin\train_omp.exe
echo.
echo Para ejecutar con diferentes hilos:
echo   set OMP_NUM_THREADS=1 ^&^& .\bin\train_omp.exe
echo   set OMP_NUM_THREADS=2 ^&^& .\bin\train_omp.exe
echo   set OMP_NUM_THREADS=4 ^&^& .\bin\train_omp.exe
echo   set OMP_NUM_THREADS=8 ^&^& .\bin\train_omp.exe
echo.
goto end

:error
echo.
echo ============================================
echo  ERROR EN COMPILACION
echo ============================================
echo.
echo Verifica que GCC este instalado con soporte OpenMP:
echo   gcc --version
echo   gcc -fopenmp --version
echo.
echo Si no esta instalado, consulta:
echo   ..\docs\INSTALL_C_TOOLS.md
echo.
exit /b 1

:end
