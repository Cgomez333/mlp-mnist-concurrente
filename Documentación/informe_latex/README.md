# Compilación del Informe LaTeX

## Requisitos

- **TeX Live** (Windows/Linux) o **MiKTeX** (Windows)
- **Overleaf** (alternativa online)

## Compilación Local

### Windows (MiKTeX)

```bash
cd Documentación/informe_latex
pdflatex main.tex
pdflatex main.tex  # Segunda pasada para referencias
```

### Linux/Mac (TeX Live)

```bash
cd Documentación/informe_latex
pdflatex main.tex
pdflatex main.tex
```

## Compilación en Overleaf

1. Ve a [Overleaf](https://www.overleaf.com/)
2. Crear nuevo proyecto → Upload Project
3. Sube `main.tex` y archivos auxiliares
4. Click en **Compile** (Ctrl+S)

## Estructura de Archivos

```
informe_latex/
├── main.tex           # Documento principal
├── README.md          # Este archivo
├── logo_ucaldas.png   # Logo de la universidad (agregar)
└── figuras/           # Carpeta para gráficas (opcional)
    ├── speedup_chart.png
    ├── accuracy_comparison.png
    └── architecture_diagram.png
```

## Notas Importantes

1. **Logo**: Agrega `logo_ucaldas.png` en la misma carpeta que `main.tex`
2. **Datos pendientes**: Completa las secciones marcadas con `[COMPLETAR]`:
   - Especificaciones de hardware (CPU, RAM, GPU)
   - Resultados de PyCUDA cuando termines la ejecución en Colab
3. **Gráficas** (opcional): Puedes agregar gráficas en la carpeta `figuras/`

## Personalización

### Cambiar autor y fecha

Edita las líneas en `main.tex`:

```latex
\author{
    Tu Nombre Completo\\
    \texttt{tu.email@ucaldas.edu.co}\\[0.5cm]
    ...
}
\date{Diciembre 2024}
```

### Agregar gráficas

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figuras/speedup_chart.png}
\caption{Comparación de speedups}
\label{fig:speedup}
\end{figure}
```

## Exportar a PDF

El comando `pdflatex` generará `main.pdf` automáticamente.

## Problemas Comunes

1. **Paquete faltante**: MiKTeX instalará automáticamente, o usa `tlmgr install <paquete>` en TeX Live
2. **Referencias rotas**: Ejecuta `pdflatex` dos veces
3. **Imágenes no aparecen**: Verifica rutas relativas

## Entrega Final

Genera el PDF final y entrega:

- ✅ `main.pdf` - Informe técnico
- ✅ Presentación PowerPoint/PDF
- ✅ Código fuente (repositorio GitHub)
