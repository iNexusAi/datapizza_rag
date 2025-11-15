"""
RAG System - Streamlit UI (DEMO VERSION - In-Memory Only)
⚠️ Questa versione NON persiste i dati su disco!
I documenti vengono persi quando si riavvia Streamlit.
Perfetta per demo e testing rapidi.
"""

import os
import sys

# Fix per certificati SSL su macOS con Homebrew Python
if sys.platform == 'darwin':  # macOS
    os.environ['SSL_CERT_FILE'] = '/opt/homebrew/etc/openssl@3/cert.pem'
    os.environ['REQUESTS_CA_BUNDLE'] = '/opt/homebrew/etc/openssl@3/cert.pem'

import streamlit as st
from rag_logic_demo import (  # ← USA LA VERSIONE DEMO!
    RAGSystem,
    process_uploaded_files,
    get_vector_size
)

# Configurazione della pagina
st.set_page_config(
    page_title="DataPizza RAG System",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carica il CSS esterno
def load_css():
    """Carica il file CSS dalla cartella static"""
    css_file = "static/style.css"
    try:
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ File CSS non trovato: {css_file}")

# Applica lo stile
load_css()

# Titolo principale
st.markdown('<h1 class="main-header">🍕 DataPizza RAG System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Retrieval-Augmented Generation con datapizza.ai</p>', unsafe_allow_html=True)
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
# Inizializzazione dello stato della sessione
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "my_documents"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Sidebar per configurazione
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Configurazione</h2>', unsafe_allow_html=True)
    
    # API Key OpenAI
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Inserisci la tua API key di OpenAI"
    )
    
    st.markdown("---")
    
    # Configurazione Qdrant
    st.markdown("### 🗄️ Qdrant Configuration")
    use_memory = st.checkbox("Usa Qdrant in-memory", value=True, help="Se selezionato, usa Qdrant in memoria (nessun server richiesto)")
    
    if not use_memory:
        qdrant_host = st.text_input("Qdrant Host", value="localhost")
        qdrant_port = st.number_input("Qdrant Port", value=6333, min_value=1, max_value=65535)
    else:
        qdrant_host = "localhost"
        qdrant_port = 6333
    
    st.markdown("---")
    
    # Configurazione modello
    st.markdown("### 🤖 Modello")
    model_name = st.selectbox(
        "Modello LLM",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="Seleziona il modello OpenAI da utilizzare"
    )
    
    embedding_model = st.selectbox(
        "Modello Embedding",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        help="Seleziona il modello di embedding da utilizzare"
    )
    
    k_documents = st.slider(
        "Numero documenti da recuperare (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Quanti documenti recuperare dal vectorstore"
    )
    
    st.markdown("---")
    
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📄 Caricamento Documenti</h2>', unsafe_allow_html=True)
    
    # Upload documenti
    uploaded_files = st.file_uploader(
        "Carica i tuoi documenti (PDF o TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Carica uno o più file PDF o TXT da indicizzare"
    )
    
    if st.button("🚀 Indicizza Documenti", disabled=not openai_api_key or not uploaded_files):
        if not openai_api_key:
            st.error("⚠️ Inserisci una API key di OpenAI nella sidebar!")
        elif not uploaded_files:
            st.error("⚠️ Carica almeno un documento!")
        else:
            with st.spinner("📚 Elaborazione documenti in corso..."):
                try:
                    # Inizializza il sistema RAG
                    st.session_state.rag_system = RAGSystem(
                        openai_api_key=openai_api_key,
                        model_name=model_name,
                        embedding_model=embedding_model
                    )
                    
                    # Inizializza Qdrant
                    st.session_state.rag_system.initialize_qdrant(
                        use_memory=use_memory,
                        host=qdrant_host,
                        port=qdrant_port
                    )
                    
                    # Crea collection
                    vector_size = get_vector_size(embedding_model)
                    st.session_state.rag_system.create_collection_if_not_exists(
                        st.session_state.collection_name,
                        vector_size
                    )
                    
                    # Processa i file
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        chunks = process_uploaded_files([uploaded_file])
                        all_chunks.extend(chunks)
                    
                    # Indicizza con progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Indicizzazione: {current}/{total} chunks")
                    
                    num_indexed = st.session_state.rag_system.index_documents(
                        st.session_state.collection_name,
                        all_chunks,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Crea la pipeline
                    st.session_state.pipeline = st.session_state.rag_system.create_pipeline(
                        collection_name=st.session_state.collection_name,
                        k=k_documents
                    )
                    
                    st.session_state.documents_loaded = True
                    
                    st.success(f"✅ Indicizzati {num_indexed} chunks da {len(uploaded_files)} documento/i!")
                    
                except Exception as e:
                    st.error(f"❌ Errore durante l'indicizzazione: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Mostra stato
    if st.session_state.documents_loaded:

        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="sub-header">💬 Chat RAG</h2>', unsafe_allow_html=True)
    
    # Verifica se ci sono documenti in Qdrant (anche da sessioni precedenti)
    documents_available = False
    if openai_api_key:
        try:
            # Prova a inizializzare il sistema RAG se non esiste
            if st.session_state.rag_system is None:
                st.session_state.rag_system = RAGSystem(
                    openai_api_key=openai_api_key,
                    model_name=model_name,
                    embedding_model=embedding_model
                )
                st.session_state.rag_system.initialize_qdrant(
                    use_memory=use_memory,
                    host=qdrant_host,
                    port=qdrant_port
                )
            
            # Verifica se la collection esiste
            if st.session_state.rag_system and st.session_state.rag_system.qdrant_client:
                try:
                    collection_info = st.session_state.rag_system.qdrant_client.get_collection(st.session_state.collection_name)
                    if collection_info.points_count > 0:
                        documents_available = True
                        if not st.session_state.documents_loaded:
                            st.info("📚 Documenti trovati da sessioni precedenti. Puoi fare domande!")
                except:
                    pass
        except:
            pass
    
    # Container per i messaggi (invertito per mostrare i più recenti in alto)
    chat_container = st.container()
    
    with chat_container:
        # Mostra i messaggi dal più recente al più vecchio
        for message in reversed(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Fonti"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Fonte {i}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
    
    # Input per la query (attivo se ci sono documenti O se API key è presente)
    chat_enabled = (st.session_state.documents_loaded or documents_available) and openai_api_key
    
    if not openai_api_key:
        st.warning("⚠️ Inserisci una API key di OpenAI nella sidebar per abilitare la chat!")
    elif not chat_enabled and not documents_available:
        st.warning("⚠️ Carica dei documenti prima di iniziare la chat!")
    
    user_query = st.chat_input("Fai una domanda sui tuoi documenti...", disabled=not chat_enabled)
    
    if user_query:
        if not openai_api_key:
            st.error("⚠️ Inserisci una API key di OpenAI nella sidebar!")
        elif not (st.session_state.documents_loaded or documents_available):
            st.error("⚠️ Prima indicizza dei documenti!")
        else:
            # Aggiungi messaggio utente in testa
            st.session_state.messages.insert(0, {"role": "user", "content": user_query})
            
            # Mostra messaggio utente
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Genera risposta con streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                sources = []
                
                try:
                    # Crea/ricrea la pipeline se necessario
                    if st.session_state.pipeline is None:
                        st.session_state.pipeline = st.session_state.rag_system.create_pipeline(
                            collection_name=st.session_state.collection_name,
                            k=k_documents
                        )
                    
                    # Usa streaming
                    for chunk_text, chunk_sources in st.session_state.rag_system.query_stream(
                        pipeline=st.session_state.pipeline,
                        user_query=user_query,
                        collection_name=st.session_state.collection_name,
                        k=k_documents
                    ):
                        full_response += chunk_text
                        message_placeholder.markdown(full_response + "▌")
                        
                        # Salva le fonti dal primo chunk
                        if chunk_sources is not None:
                            sources = chunk_sources
                    
                    # Mostra risposta finale senza cursore
                    message_placeholder.markdown(full_response)
                    
                    # Mostra fonti
                    if sources:
                        with st.expander("📚 Fonti"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Fonte {i}:**")
                                st.text(source[:300] + "..." if len(source) > 300 else source)
                    
                    # Salva nella cronologia (in testa)
                    st.session_state.messages.insert(0, {
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_message = f"❌ Errore durante la generazione della risposta: {str(e)}"
                    st.error(error_message)
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.messages.insert(0, {
                        "role": "assistant",
                        "content": error_message
                    })
            
            # Forza rerun per aggiornare la visualizzazione
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Powered by <strong>datapizza.ai</strong> 🍕 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

