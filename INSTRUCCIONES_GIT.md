# 🔧 INSTRUCCIONES: Cómo Gestionar Git y Traer Código de devS

## 🎯 Situación Actual

**Tu rama `dev`**:

- Tienes cambios sin commitear (frontend, API, exportación mejorada)
- Estás 1 commit adelante de `origin/dev`

**Rama `devS` (compañero)**:

- Tiene código Python que necesitas
- Tiene estructura diferente (movió carpetas)

---

## ✅ PLAN RECOMENDADO (Paso a Paso)

### PASO 1: Guardar tu trabajo actual

```bash
cd "c:\Users\carli\OneDrive\Desktop\Universidad de Caldas\Semestre VII\Concurrentes\Proyecto\mlp-mnist-concurrente"

# Ver qué cambios tienes
git status

# Agregar todos los cambios
git add .

# Commitear
git commit -m "feat: Frontend React + API Node.js + exportación de pesos mejorada

- Agregado frontend React con Vite
- Refactorizada API para servir múltiples modelos
- Mejorada exportación de pesos en C (sequential y openmp)
- Agregado visualizador de MNIST
"
```

### PASO 2: Pushear a tu rama

```bash
git push origin dev
```

**✅ Ahora tu trabajo está seguro en GitHub**

### PASO 3: Traer código Python sin romper tu estructura

**Opción A: Cherry-pick (RECOMENDADA)**

Esta opción trae SOLO los archivos de Python sin tocar tu estructura:

```bash
# Crear una nueva rama de trabajo (por seguridad)
git checkout -b dev-integration

# Traer solo las carpetas de Python de devS
git checkout origin/devS -- py_secuencial
git checkout origin/devS -- py_multiprocessing

# Commitear la integración
git commit -m "feat: Integrar implementaciones Python desde devS"

# Volver a dev y hacer merge
git checkout dev
git merge dev-integration

# Pushear
git push origin dev

# Eliminar rama temporal
git branch -d dev-integration
```

**Opción B: Merge completo (puede dar conflictos)**

```bash
# Hacer merge de devS en dev
git merge origin/devS -m "merge: Integrar código Python desde devS"

# Si hay conflictos, Git te mostrará:
# CONFLICT (content): Merge conflict in <archivo>

# Para cada conflicto, edita el archivo y elige qué versión mantener:
# <<<<<<< HEAD
#   Tu código
# =======
#   Código de devS
# >>>>>>> origin/devS

# Después de resolver:
git add <archivo-resuelto>
git commit -m "merge: Resueltos conflictos de integración"
git push origin dev
```

---

## 🚨 ADVERTENCIA: Conflictos Esperados

Si usas Opción B (merge), estos archivos probablemente tendrán conflictos:

1. `README.md` - Ambos lo modificaron
2. Rutas de carpetas - devS movió `c_*` a la raíz
3. `backend/data/mnist/*` - Son archivos binarios grandes (Git NO los versiona)

**Solución**: Usa `.gitignore` para excluir datos:

```bash
# Crear/editar .gitignore
echo "backend/data/mnist/*.bin" >> .gitignore
echo "backend/data/mnist/*-ubyte" >> .gitignore
git add .gitignore
git commit -m "chore: Ignorar archivos binarios de MNIST"
```

---

## 📁 Estructura Final Esperada

Después de la integración, deberías tener:

```
mlp-mnist-concurrente/
├── backend/
│   ├── api/                    # TU CÓDIGO (Node.js)
│   ├── c_secuencial/           # TU CÓDIGO (C)
│   ├── c_openmp/               # TU CÓDIGO (C + OpenMP)
│   ├── data/                   # Dataset (NO se versiona)
│   ├── docs/                   # Documentación
│   ├── scripts/                # Scripts Python compartidos
│   └── results/                # Resultados (CSV, pesos)
├── frontend/                   # TU CÓDIGO (React)
├── py_secuencial/              # CÓDIGO DEVS (Python)
├── py_multiprocessing/         # CÓDIGO DEVS (Python)
├── GUIA_COMPLETA_PROYECTO.md   # Esta guía
└── README.md                   # Actualizar con toda la info
```

---

## ✅ Verificación Post-Integración

```bash
# Verificar que todo esté
ls py_secuencial/
ls py_multiprocessing/

# Verificar que tu código siga intacto
ls frontend/
ls backend/api/

# Ver historial de commits
git log --oneline --graph --all -10
```

---

## 🆘 Si Algo Sale Mal

### Deshacer cambios NO commiteados

```bash
git restore <archivo>
```

### Volver al estado anterior al merge

```bash
git merge --abort
```

### Volver al último commit

```bash
git reset --hard HEAD
```

### Recuperar trabajo perdido (si lo commiteaste antes)

```bash
git reflog  # Ver historial de todas las acciones
git checkout <hash-del-commit>
```

---

## 💡 RECOMENDACIÓN FINAL

**Usa la Opción A (cherry-pick)** porque:

- ✅ Más seguro (no tocas tu estructura)
- ✅ Solo traes lo que necesitas
- ✅ Evitas conflictos innecesarios
- ✅ Mantienes control total

**Evita la Opción B (merge completo)** porque:

- ⚠️ devS tiene estructura diferente (carpetas movidas)
- ⚠️ Puede sobrescribir tu frontend/API
- ⚠️ Requiere resolver conflictos manualmente

---

## 📞 Checklist Pre-Push

Antes de hacer `git push`, verifica:

- [ ] `git status` muestra "working tree clean"
- [ ] `git log` muestra tu commit de integración
- [ ] Frontend sigue funcionando: `cd frontend && npm run dev`
- [ ] API sigue funcionando: `cd backend/api && npm start`
- [ ] Código Python está presente: `ls py_secuencial`

---

## 🎓 Para Aprender Más

```bash
# Ver ramas locales y remotas
git branch -a

# Ver diferencias entre ramas
git diff dev..origin/devS

# Ver archivos cambiados entre ramas
git diff dev..origin/devS --name-only

# Ver commits únicos de cada rama
git log dev..origin/devS --oneline
git log origin/devS..dev --oneline
```
