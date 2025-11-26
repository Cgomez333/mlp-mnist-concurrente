@echo off
echo ============================================
echo  MLP MNIST - Compilacion C Secuencial
echo ============================================
echo.

REM Crear directorios
if not exist build mkdir build
if not exist bin mkdir bin
if not exist ..\results mkdir ..\results
if not exist ..\results\raw mkdir ..\results\raw

echo [1/5] Compilando data.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/data.c -o build/data.o
if errorlevel 1 goto error

echo [2/5] Compilando matrix.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/matrix.c -o build/matrix.o
if errorlevel 1 goto error

echo [3/5] Compilando mlp.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/mlp.c -o build/mlp.o
if errorlevel 1 goto error

echo [4/5] Compilando train.c...
gcc -Wall -Wextra -O3 -Iinclude -c src/train.c -o build/train.o
if errorlevel 1 goto error

echo [5/5] Enlazando...
gcc build/data.o build/matrix.o build/mlp.o build/train.o -lm -o bin/train_seq.exe
if errorlevel 1 goto error

echo.
echo ============================================
echo  COMPILACION EXITOSA
echo ============================================
echo.
echo Ejecutable: bin\train_seq.exe
echo.
echo Para ejecutar:
echo   .\bin\train_seq.exe
echo.
goto end

:error
echo.
echo ============================================
echo  ERROR EN COMPILACION
echo ============================================
echo.
echo Verifica que GCC este instalado:
echo   gcc --version
echo.
echo Si no esta instalado, consulta:
echo   ..\docs\INSTALL_C_TOOLS.md
echo.
exit /b 1

:end
