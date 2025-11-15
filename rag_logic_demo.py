"""
RAG Logic Module - Demo Version (In-Memory Only)
Versione demo che NON persiste i dati su disco.
I dati vengono persi quando si riavvia Streamlit.
Utile per demo e testing rapidi.
"""

import os
import sys

# Fix per certificati SSL su macOS con Homebrew Python
if sys.platform == 'darwin':  # macOS
    os.environ['SSL_CERT_FILE'] = '/opt/homebrew/etc/openssl@3/cert.pem'
    os.environ['REQUESTS_CA_BUNDLE'] = '/opt/homebrew/etc/openssl@3/cert.pem'

from PyPDF2 import PdfReader
from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class RAGSystem:
    """Classe principale per gestire il sistema RAG - Versione In-Memory"""
    
    def __init__(self, openai_api_key, model_name="gpt-4o-mini", embedding_model="text-embedding-3-small"):
        """
        Inizializza il sistema RAG
        
        Args:
            openai_api_key: API key di OpenAI
            model_name: Nome del modello LLM da usare
            embedding_model: Nome del modello di embedding da usare
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.qdrant_client = None
        self.use_memory = True  # Sempre in-memory per la versione demo
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        
    def initialize_qdrant(self, use_memory=True, host="localhost", port=6333):
        """
        Inizializza il client Qdrant IN-MEMORY ONLY
        
        Args:
            use_memory: Ignorato - sempre True nella versione demo
            host: Host di Qdrant (se use_memory=False) - non usato nella demo
            port: Porta di Qdrant (se use_memory=False) - non usato nella demo
            
        Returns:
            QdrantClient instance
        """
        # VERSIONE DEMO: Usa SEMPRE :memory: (nessuna persistenza)
        self.use_memory = True  # Forza sempre in-memory
        self.qdrant_host = host
        self.qdrant_port = port
        
        # Usa Qdrant completamente in RAM
        self.qdrant_client = QdrantClient(":memory:")
        print("🧪 Qdrant inizializzato in modalità IN-MEMORY (Demo)")
        print("⚠️  I dati verranno persi al riavvio di Streamlit!")
        
        return self.qdrant_client
    
    def create_collection_if_not_exists(self, collection_name, vector_size=1536):
        """
        Crea una collection se non esiste
        
        Args:
            collection_name: Nome della collection
            vector_size: Dimensione dei vettori (1536 per small/ada, 3072 per large)
        """
        if not self.qdrant_client:
            raise ValueError("Qdrant client non inizializzato. Chiama initialize_qdrant() prima.")
        
        try:
            # Prova a ottenere la collection esistente
            existing_collection = self.qdrant_client.get_collection(collection_name)
            # Se esiste, cancellala per ricrearla con la struttura corretta
            self.qdrant_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' cancellata, verrà ricreata con la struttura corretta")
        except:
            # La collection non esiste, va bene
            pass
        
        # Crea la collection con un nome esplicito per il vettore
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(size=vector_size, distance=Distance.COSINE)
            }
        )
    
    def index_documents(self, collection_name, chunks, progress_callback=None):
        """
        Indicizza i documenti nel vectorstore
        
        Args:
            collection_name: Nome della collection
            chunks: Lista di chunk di testo da indicizzare
            progress_callback: Funzione callback per aggiornare il progresso (opzionale)
            
        Returns:
            Numero di chunk indicizzati
        """
        if not self.qdrant_client:
            raise ValueError("Qdrant client non inizializzato. Chiama initialize_qdrant() prima.")
        
        # Inizializza embedder
        embedder = OpenAIEmbedder(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        
        points = []
        
        for i, chunk in enumerate(chunks):
            # Genera embedding
            embedding_result = embedder.run(text=chunk)
            
            # L'embedder restituisce direttamente una lista o un dict con 'embedding'
            if isinstance(embedding_result, list):
                embedding = embedding_result
            elif isinstance(embedding_result, dict):
                embedding = embedding_result.get("embedding", embedding_result.get("vector", []))
            else:
                raise ValueError(f"Formato embedding non valido: {type(embedding_result)}")
            
            # Crea point per Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"default": embedding},  # Specifica il nome del vettore
                payload={"text": chunk}
            )
            points.append(point)
            
            # Callback per progress bar
            if progress_callback:
                progress_callback(i + 1, len(chunks))
        
        # Carica i points in Qdrant
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return len(points)
    
    def create_pipeline(self, collection_name, k=3):
        """
        Crea la pipeline RAG completa
        
        Args:
            collection_name: Nome della collection Qdrant
            k: Numero di documenti da recuperare
            
        Returns:
            DagPipeline configurata
        """
        # Inizializza componenti
        openai_client = OpenAIClient(
            model=self.model_name,
            api_key=self.openai_api_key
        )
        
        embedder = OpenAIEmbedder(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        
        # VERSIONE DEMO: Usa sempre il client in-memory
        vectorstore = QdrantVectorstore(host="localhost", port=6333)
        vectorstore.client = self.qdrant_client  # Usa il client in-memory
        
        # Crea pipeline
        dag_pipeline = DagPipeline()
        dag_pipeline.add_module(
            "rewriter", 
            ToolRewriter(
                client=openai_client, 
                system_prompt="Riscrivi le query dell'utente per migliorare l'accuratezza del recupero."
            )
        )
        dag_pipeline.add_module("embedder", embedder)
        dag_pipeline.add_module(
            "retriever", 
            vectorstore.as_retriever(
                collection_name=collection_name, 
                k=k,
                vector_name="default"  # Specifica il nome del vettore
            )
        )
        dag_pipeline.add_module(
            "prompt", 
            ChatPromptTemplate(
                user_prompt_template="Domanda dell'utente: {{user_prompt}}\n",
                retrieval_prompt_template="Contenuto recuperato:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
            )
        )
        dag_pipeline.add_module("generator", openai_client)
        
        # Connetti i moduli
        dag_pipeline.connect("rewriter", "embedder", target_key="text")
        dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
        dag_pipeline.connect("retriever", "prompt", target_key="chunks")
        dag_pipeline.connect("prompt", "generator", target_key="memory")
        
        return dag_pipeline
    
    def query(self, pipeline, user_query, collection_name, k=3):
        """
        Esegue una query sulla pipeline RAG
        
        Args:
            pipeline: DagPipeline configurata
            user_query: Query dell'utente
            collection_name: Nome della collection
            k: Numero di documenti da recuperare
            
        Returns:
            Tuple (response, sources) dove:
                - response: La risposta generata
                - sources: Lista dei chunk recuperati
        """
        result = pipeline.run({
            "rewriter": {"user_prompt": user_query},
            "prompt": {"user_prompt": user_query},
            "retriever": {"collection_name": collection_name, "k": k},
            "generator": {"input": user_query}
        })
        
        # Estrai risposta e fonti
        raw_response = result['generator']
        
        # Il generatore può restituire diversi formati
        if isinstance(raw_response, str):
            # È già una stringa
            response = raw_response
        elif hasattr(raw_response, 'content'):
            # È un oggetto ClientResponse con content
            if isinstance(raw_response.content, list) and len(raw_response.content) > 0:
                # content è una lista di TextBlock
                text_blocks = []
                for block in raw_response.content:
                    if hasattr(block, 'content'):
                        text_blocks.append(block.content)
                    elif hasattr(block, 'text'):
                        text_blocks.append(block.text)
                response = '\n'.join(text_blocks)
            elif isinstance(raw_response.content, str):
                response = raw_response.content
            else:
                response = str(raw_response.content)
        elif hasattr(raw_response, 'text'):
            # Ha un attributo text
            response = raw_response.text
        elif hasattr(raw_response, 'message'):
            # Ha un attributo message
            if hasattr(raw_response.message, 'content'):
                response = raw_response.message.content
            else:
                response = str(raw_response.message)
        else:
            # Fallback: converti a stringa
            response = str(raw_response)
        
        retrieved_chunks = result.get('retriever', [])
        
        # I chunk possono essere oggetti Chunk o dizionari
        sources = []
        for chunk in retrieved_chunks:
            if hasattr(chunk, 'text'):
                # È un oggetto Chunk con attributo text
                sources.append(chunk.text)
            elif isinstance(chunk, dict) and 'text' in chunk:
                # È un dizionario con chiave 'text'
                sources.append(chunk['text'])
            elif hasattr(chunk, 'payload') and isinstance(chunk.payload, dict):
                # È un oggetto con payload che contiene text
                sources.append(chunk.payload.get('text', ''))
        
        return response, sources
    
    def query_stream(self, pipeline, user_query, collection_name, k=3):
        """
        Esegue una query sulla pipeline RAG con streaming della risposta
        
        Args:
            pipeline: DagPipeline configurata
            user_query: Query dell'utente
            collection_name: Nome della collection
            k: Numero di documenti da recuperare
            
        Yields:
            Tuple (chunk_text, sources) dove:
                - chunk_text: Chunk di testo della risposta (streaming)
                - sources: Lista dei chunk recuperati (solo nel primo yield)
        """
        # Prima esegui retrieval e rewriting
        from datapizza.clients.openai import OpenAIClient
        from datapizza.embedders.openai import OpenAIEmbedder
        from datapizza.modules.rewriters import ToolRewriter
        
        # Inizializza componenti per le operazioni preliminari
        openai_client = OpenAIClient(
            model=self.model_name,
            api_key=self.openai_api_key
        )
        
        embedder = OpenAIEmbedder(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        
        rewriter = ToolRewriter(
            client=openai_client,
            system_prompt="Riscrivi le query dell'utente per migliorare l'accuratezza del recupero."
        )
        
        # Rewrite query
        rewritten = rewriter.run(user_prompt=user_query)
        rewritten_text = rewritten.get("text", user_query) if isinstance(rewritten, dict) else user_query
        
        # Generate embedding
        embedding_result = embedder.run(text=rewritten_text)
        if isinstance(embedding_result, list):
            query_vector = embedding_result
        elif isinstance(embedding_result, dict):
            query_vector = embedding_result.get("embedding", embedding_result.get("vector", []))
        else:
            query_vector = []
        
        # Retrieve documents - usa sempre client in-memory
        vectorstore_retriever = QdrantVectorstore(host="localhost", port=6333)
        vectorstore_retriever.client = self.qdrant_client
        
        retriever = vectorstore_retriever.as_retriever(
            collection_name=collection_name,
            k=k,
            vector_name="default"
        )
        
        retrieved_chunks = retriever.run(query_vector=query_vector, collection_name=collection_name, k=k)
        
        # Extract sources
        sources = []
        for chunk in retrieved_chunks:
            if hasattr(chunk, 'text'):
                sources.append(chunk.text)
            elif isinstance(chunk, dict) and 'text' in chunk:
                sources.append(chunk['text'])
            elif hasattr(chunk, 'payload') and isinstance(chunk.payload, dict):
                sources.append(chunk.payload.get('text', ''))
        
        # Build context
        context = f"Domanda dell'utente: {user_query}\n\nContenuto recuperato:\n"
        for chunk_text in sources:
            context += f"{chunk_text}\n"
        
        # Stream response
        first_chunk = True
        for chunk in openai_client.stream_invoke(context):
            if chunk.delta:
                if first_chunk:
                    # Nel primo chunk, invia anche le fonti
                    yield chunk.delta, sources
                    first_chunk = False
                else:
                    yield chunk.delta, None


# Funzioni helper per il processing dei documenti
# (Identiche alla versione persistente)

def extract_text_from_pdf(pdf_file):
    """
    Estrae il testo da un file PDF
    
    Args:
        pdf_file: File PDF (file-like object)
        
    Returns:
        Testo estratto dal PDF
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Divide il testo in chunk con overlap
    
    Args:
        text: Testo da dividere
        chunk_size: Dimensione di ogni chunk
        overlap: Numero di caratteri di sovrapposizione tra i chunk
        
    Returns:
        Lista di chunk di testo
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    
    return chunks


def process_uploaded_files(uploaded_files):
    """
    Processa una lista di file caricati (PDF o TXT)
    
    Args:
        uploaded_files: Lista di file caricati
        
    Returns:
        Lista di tutti i chunk estratti dai file
    """
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Estrai testo
        if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
        
        # Chunking
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    return all_chunks


def get_vector_size(embedding_model):
    """
    Restituisce la dimensione dei vettori per un dato modello di embedding
    
    Args:
        embedding_model: Nome del modello di embedding
        
    Returns:
        Dimensione dei vettori
    """
    if "small" in embedding_model or "ada" in embedding_model:
        return 1536
    elif "large" in embedding_model:
        return 3072
    else:
        return 1536  # default

