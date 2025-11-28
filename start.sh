#!/bin/bash

# Script para iniciar TODO el proyecto (Frontend + Backend API)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║       🚀 Iniciando MLP MNIST - Proyecto Completo          ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para manejar Ctrl+C
function cleanup {
    echo ""
    echo "${YELLOW}⚠️  Deteniendo servidores...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "${GREEN}✅ Servidores detenidos${NC}"
    exit 0
}

trap cleanup INT TERM

# Iniciar Backend API
echo "${BLUE}📡 Iniciando Backend API (Puerto 3001)...${NC}"
cd backend/api
npm start > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "${GREEN}✅ Backend iniciado (PID: $BACKEND_PID)${NC}"
cd ../..

# Esperar 2 segundos
sleep 2

# Iniciar Frontend React
echo ""
echo "${BLUE}🎨 Iniciando Frontend React (Puerto 5173)...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "${GREEN}✅ Frontend iniciado (PID: $FRONTEND_PID)${NC}"
cd ..

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║       ✅ TODO ESTÁ CORRIENDO                               ║"
echo "║                                                            ║"
echo "║       Frontend:  http://localhost:5173                     ║"
echo "║       Backend:   http://localhost:3001                     ║"
echo "║                                                            ║"
echo "║       Logs en:   logs/backend.log                          ║"
echo "║                  logs/frontend.log                         ║"
echo "║                                                            ║"
echo "║       Presiona Ctrl+C para detener todo                    ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Mantener el script corriendo
wait
