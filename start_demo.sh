#!/bin/bash

# Script per avviare l'applicazione RAG - VERSIONE DEMO (In-Memory)

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Avvio DataPizza RAG System - VERSIONE DEMO (In-Memory)...${NC}"
echo -e "${YELLOW}⚠️  I dati NON verranno salvati su disco!${NC}"
echo ""

# Verifica se l'ambiente virtuale è attivo
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}⚠️  Ambiente virtuale non attivo!${NC}"
    echo -e "${BLUE}Attivazione in corso...${NC}"
    source env_project/bin/activate
fi

# Verifica se streamlit è installato
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}⚠️  Streamlit non trovato!${NC}"
    echo -e "${BLUE}Installazione dipendenze in corso...${NC}"
    pip install -r requirements.txt
fi

echo -e "${GREEN}✅ Avvio dell'applicazione DEMO...${NC}"
echo ""

# Avvia streamlit con la versione demo
streamlit run run_demo.py

echo ""
echo -e "${GREEN}✅ Applicazione terminata.${NC}"

