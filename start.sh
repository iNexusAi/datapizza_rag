#!/bin/bash

# Script per avviare l'applicazione RAG

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🍕 Avvio DataPizza RAG System...${NC}"
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

echo -e "${GREEN}✅ Avvio dell'applicazione...${NC}"
echo ""

# Avvia streamlit
streamlit run run.py

echo ""
echo -e "${GREEN}✅ Applicazione terminata.${NC}"

