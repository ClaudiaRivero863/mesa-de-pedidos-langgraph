"""Configuracion central: rutas y modelos. Los modelos se pueden cambiar desde .env."""
import os
from pathlib import Path
 
from dotenv import load_dotenv
 
RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")  # carga OPENAI_API_KEY y los opcionales
 
# --- Rutas
DIR_DATOS = RAIZ / "datos"
DIR_POLITICAS = DIR_DATOS / "politicas"
DIR_ARTEFACTOS = RAIZ / "artefactos"
 
# --- Modelos (valores por defecto: los del .env.example de la catedra)
MODELO_LLM = os.getenv("MODELO_LLM", "gpt-4o-mini")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "text-embedding-3-small")
 
# --- Retriever: denso | bm25 | hibrido (se cambia con RETRIEVER en el .env)
RETRIEVER = os.getenv("RETRIEVER", "hibrido")
 
# --- Descomposicion de preguntas compuestas (1 = si, 0 = no). Se cambia con DESCOMPONER en el .env
DESCOMPONER = os.getenv("DESCOMPONER", "1") == "1"
 
# --- Chunking (decision propia: la consigna no lo fija)
# "articulos": un fragmento por articulo del PDF (o por SKU en las fichas).
# "recursivo": corta cada pagina cada CHUNK_SIZE caracteres, sin mirar el contenido.
# Cada estrategia tiene su propio indice: artefactos/indice_faiss_<estrategia>.
ESTRATEGIA_CHUNKING = os.getenv("ESTRATEGIA_CHUNKING", "articulos")
CHUNK_SIZE = 800  # solo para "recursivo"
CHUNK_OVERLAP = 100  # solo para "recursivo"
CHUNK_MAX_ARTICULO = 1200  # un articulo mas largo que esto se parte en pedazos
 
DIR_INDICE = DIR_ARTEFACTOS / f"indice_faiss_{ESTRATEGIA_CHUNKING}"
 