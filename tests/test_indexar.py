"""Pruebas del chunking por articulo. Usan los PDF reales pero NO la API."""
import pytest
from langchain_core.documents import Document
 
from mesa_pedidos import config, indexar
 
 
@pytest.fixture(scope="module")
def fragmentos():
    return indexar.partir_por_articulos(indexar.cargar_paginas())
 
 
def de(fragmentos, documento, empieza):
    return [d for d in fragmentos
            if d.metadata["documento"] == documento and d.metadata["seccion"].startswith(empieza)]
 
 
def test_el_articulo_de_devoluciones_queda_entero_en_un_fragmento(fragmentos):
    """El caso MSG-003: la regla 'zapatos de seguridad no se devuelven' va junto a su titulo."""
    (art2,) = de(fragmentos, "politica_devoluciones_2026.pdf", "2. DEVOLUCION")
    assert "NO se aceptan devoluciones" in art2.page_content
    assert "zapatos de seguridad" in art2.page_content
    assert "15 dias corridos" in art2.page_content
    assert art2.metadata["pagina"] == 1
 
 
def test_la_escala_de_descuentos_esta_completa_en_un_fragmento(fragmentos):
    (art4,) = de(fragmentos, "condiciones_comerciales_2026.pdf", "ART. 4")
    for escalon in ("0%", "5%", "8%", "12%"):
        assert escalon in art4.page_content
 
 
def test_un_articulo_que_sigue_en_la_pagina_siguiente_conserva_su_pagina(fragmentos):
    art5 = de(fragmentos, "condiciones_comerciales_2026.pdf", "ART. 5")
    assert [d.metadata["pagina"] for d in art5] == [1, 2]
    assert "(continuacion)" in art5[1].page_content
    assert "20%" in art5[0].page_content and "15%" in art5[1].page_content
 
 
def test_hay_un_fragmento_por_sku_con_su_rubro(fragmentos):
    fichas = [d for d in fragmentos if d.metadata["documento"] == "fichas_tecnicas_2026.pdf"]
    assert len(fichas) == 40
    (he4330,) = de(fragmentos, "fichas_tecnicas_2026.pdf", "HE-4330")
    assert "HERRAMIENTA ELECTRICA" in he4330.page_content
    assert "SIN BATERIA" in he4330.page_content and "HE-4335" in he4330.page_content
 
 
def test_el_membrete_no_se_indexa(fragmentos):
    assert not any(d.page_content.startswith("FERRETERIA INDUSTRIAL ANDINA") for d in fragmentos)
 
 
def test_todos_los_fragmentos_traen_documento_pagina_y_seccion(fragmentos):
    assert all({"documento", "pagina", "seccion"} <= set(d.metadata) for d in fragmentos)
 
 
def test_un_articulo_demasiado_largo_se_parte_y_repite_el_titulo(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_ARTICULO", 200)
    largo = "ART. 1 - LARGO\n" + "\n".join(f"Linea numero {i} con texto de relleno." for i in range(30))
    pagina = Document(page_content=largo, metadata={"documento": "x.pdf", "pagina": 1})
    partes = indexar.partir_por_articulos([pagina])
    assert len(partes) > 1
    assert all(p.metadata["seccion"] == "ART. 1 - LARGO" for p in partes)
    assert all("ART. 1 - LARGO" in p.page_content for p in partes)
 
 
def test_la_estrategia_se_elige_por_config_y_se_valida(monkeypatch):
    paginas = indexar.cargar_paginas()
    monkeypatch.setattr(config, "ESTRATEGIA_CHUNKING", "recursivo")
    assert len(indexar.partir(paginas)) == 27  # el chunking anterior, por tamanio
    monkeypatch.setattr(config, "ESTRATEGIA_CHUNKING", "articulos")
    assert len(indexar.partir(paginas)) == 63
    monkeypatch.setattr(config, "ESTRATEGIA_CHUNKING", "inventada")
    with pytest.raises(ValueError):
        indexar.partir(paginas)
 
 
def test_cada_estrategia_tiene_su_propio_indice():
    assert config.DIR_INDICE.name == f"indice_faiss_{config.ESTRATEGIA_CHUNKING}"
 