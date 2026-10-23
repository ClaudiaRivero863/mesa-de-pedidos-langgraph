"""Indexa los PDF de datos/politicas/ en FAISS y persiste el indice en disco.
 
Uso (desde la raiz del proyecto):
    python -m mesa_pedidos.indexar            # indexa solo si no hay indice
    python -m mesa_pedidos.indexar --forzar   # reindexa de cero
"""
import argparse
import re
 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
 
from . import config
 
 
def cargar_paginas() -> list[Document]:
    """Una Document por pagina de cada PDF, con documento y pagina como metadata."""
    paginas = []
    for pdf in sorted(config.DIR_POLITICAS.glob("*.pdf")):
        lector = PdfReader(pdf)
        for numero, pagina in enumerate(lector.pages, start=1):
            texto = (pagina.extract_text() or "").strip()
            if texto:
                paginas.append(
                    Document(
                        page_content=texto,
                        metadata={"documento": pdf.name, "pagina": numero},
                    )
                )
    return paginas
 
 
# Un encabezado de seccion: "ART. 4 -", "2. DEVOLUCION..." o un SKU ("AB-1020 -").
_ENCABEZADO = re.compile(r"^(ART\.\s*\d+\s*-|\d+\.\s+[A-Z]{3}|[A-Z]{2}-\d{4}\s-)")
# Una linea solo en mayusculas es un rubro de las fichas ("SEGURIDAD E HIGIENE").
_RUBRO = re.compile(r"^[A-ZÁÉÍÓÚÑ ,]{4,}$")
 
 
def partir_recursivo(paginas: list[Document]) -> list[Document]:
    """Corta cada pagina cada CHUNK_SIZE caracteres, sin mirar donde empieza un articulo."""
    cortador = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return cortador.split_documents(paginas)
 
 
def _armar(titulo: str | None, rubro: str | None, lineas: list[str], continuacion: bool,
           meta: dict) -> list[Document]:
    """Convierte un bloque de lineas en uno o mas fragmentos con su metadata."""
    if not titulo or not lineas:
        return []  # lo que va antes del primer encabezado es el membrete: no se indexa
    cuerpo = "\n".join(lineas)
    encabezado = f"{titulo} (continuacion)" if continuacion else None
    texto = "\n".join(p for p in (rubro, encabezado, cuerpo) if p)
    metadata = {**meta, "seccion": titulo}
    if len(texto) <= config.CHUNK_MAX_ARTICULO:
        return [Document(page_content=texto, metadata=metadata)]
    # Un articulo demasiado largo se parte, y cada pedazo repite el titulo para no quedar suelto.
    cortador = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_MAX_ARTICULO, chunk_overlap=config.CHUNK_OVERLAP
    )
    pedazos = cortador.split_text(texto)
    return [
        Document(page_content=p if i == 0 else f"{titulo} (continuacion)\n{p}", metadata=metadata)
        for i, p in enumerate(pedazos)
    ]
 
 
def partir_por_articulos(paginas: list[Document]) -> list[Document]:
    """Un fragmento por articulo (o por SKU), con la pagina donde esta el texto.
 
    Un articulo que sigue en la pagina siguiente genera otro fragmento que repite el titulo,
    asi la cita 'documento, p. N' sigue siendo exacta.
    """
    fragmentos: list[Document] = []
    documento, seccion, rubro = None, None, None
    for pagina in paginas:
        meta = dict(pagina.metadata)
        if meta["documento"] != documento:
            documento, seccion, rubro = meta["documento"], None, None
        titulo, cont, lineas = seccion, seccion is not None, []
        for linea in pagina.page_content.split("\n"):
            linea = linea.strip()
            if not linea:
                continue
            if _ENCABEZADO.match(linea):
                fragmentos += _armar(titulo, rubro, lineas, cont, meta)
                titulo, cont, lineas = linea, False, [linea]
                seccion = linea
            elif _RUBRO.match(linea):
                fragmentos += _armar(titulo, rubro, lineas, cont, meta)
                rubro, titulo, cont, lineas = linea, None, False, []
            else:
                lineas.append(linea)
        fragmentos += _armar(titulo, rubro, lineas, cont, meta)
    return fragmentos
 
 
def partir(paginas: list[Document]) -> list[Document]:
    """Parte las paginas en fragmentos segun config.ESTRATEGIA_CHUNKING."""
    if config.ESTRATEGIA_CHUNKING == "articulos":
        return partir_por_articulos(paginas)
    if config.ESTRATEGIA_CHUNKING == "recursivo":
        return partir_recursivo(paginas)
    raise ValueError(
        f"ESTRATEGIA_CHUNKING debe ser 'articulos' o 'recursivo', no '{config.ESTRATEGIA_CHUNKING}'"
    )
 
 
def indexar(forzar: bool = False) -> None:
    if config.DIR_INDICE.exists() and not forzar:
        print(f"El indice ya existe en {config.DIR_INDICE}. Usa --forzar para rehacerlo.")
        return
    paginas = cargar_paginas()
    fragmentos = partir(paginas)
    print(f"{len(paginas)} paginas -> {len(fragmentos)} fragmentos "
          f"(estrategia: {config.ESTRATEGIA_CHUNKING}). Generando embeddings...")
    embeddings = OpenAIEmbeddings(model=config.MODELO_EMBEDDINGS)
    indice = FAISS.from_documents(fragmentos, embeddings)
    config.DIR_INDICE.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(config.DIR_INDICE))
    print(f"Indice guardado en {config.DIR_INDICE}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forzar", action="store_true", help="reindexar aunque exista")
    indexar(parser.parse_args().forzar)
 