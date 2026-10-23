"""Pruebas del retriever (denso, BM25, hibrido). Usan los PDF reales pero NO la API:
los embeddings se reemplazan por una busqueda densa simulada."""
import pytest
from langchain_core.documents import Document
 
from mesa_pedidos import config, indexar, recuperacion as rec
from mesa_pedidos.esquemas import SubPreguntas
 
PREGUNTA_004 = ("el atornillador de impacto SKU HE-4330 viene con bateria incluida? "
                "Y si no viene, cual es el codigo de la bateria compatible?")
 
 
@pytest.fixture(scope="module")
def fragmentos():
    return indexar.partir(indexar.cargar_paginas())  # los 27 fragmentos reales
 
 
class IndiceFalso:
    """Imita al FAISS: la busqueda densa 'se equivoca' y trae fragmentos que no corresponden."""
 
    def __init__(self, fragmentos, denso):
        self.docstore = type("D", (), {"_dict": dict(enumerate(fragmentos))})()
        self._denso = denso
 
    def similarity_search(self, pregunta, k):
        return self._denso[:k]
 
 
def sin_codigo(fragmentos, codigo="HE-4330"):
    return [d for d in fragmentos if codigo not in d.page_content]
 
 
def test_tokenizar_conserva_codigos_y_numeros_y_quita_tildes():
    assert rec.tokenizar("HE-4330") == ["he-4330"]
    assert rec.tokenizar("Bateria compatible: HE-4335.") == ["bateria", "compatible", "he-4335"]
    assert rec.tokenizar("flete $175.000 por envío") == ["flete", "175.000", "por", "envio"]
 
 
def test_bm25_encuentra_el_sku_literal(fragmentos):
    top = rec._buscar_bm25(PREGUNTA_004, fragmentos, k=3)
    assert "HE-4330" in top[0].page_content
    assert top[0].metadata["documento"] == "fichas_tecnicas_2026.pdf"
 
 
def test_bm25_distingue_skus_parecidos(fragmentos):
    top = rec._buscar_bm25("HE-4410", fragmentos, k=1)
    assert "HE-4410" in top[0].page_content
 
 
def test_rrf_premia_al_fragmento_que_esta_en_los_dos_rankings():
    mk = lambda t: Document(page_content=t, metadata={"documento": "d.pdf", "pagina": 1})
    a, b, c = mk("a"), mk("b"), mk("c")
    fusion = rec.fusionar_rrf([a, b], [c, b], k=3)
    assert fusion[0].page_content == "b"  # segundo en ambos le gana a primero en uno solo
 
 
def test_hibrido_sube_lo_que_la_densa_deja_en_un_lugar_bajo(fragmentos, monkeypatch):
    """La densa ve el fragmento con HE-4330 pero lo pone septimo (afuera del top 4).
    BM25 lo pone primero por el codigo literal: la fusion lo sube al top 4."""
    con_codigo = [d for d in fragmentos if "HE-4330" in d.page_content]
    otros = sin_codigo(fragmentos)
    densa = otros[:6] + con_codigo + otros[6:]
    monkeypatch.setattr(rec, "cargar_indice", lambda: IndiceFalso(fragmentos, densa))
 
    denso = rec.buscar(PREGUNTA_004, modo="denso")
    hibrido = rec.buscar(PREGUNTA_004, modo="hibrido")
 
    assert not any("HE-4330" in d.page_content for d in denso)
    assert any("HE-4330" in d.page_content for d in hibrido)
 
 
def test_el_modo_sale_de_config_y_se_valida(fragmentos, monkeypatch):
    monkeypatch.setattr(rec, "cargar_indice", lambda: IndiceFalso(fragmentos, sin_codigo(fragmentos)))
    monkeypatch.setattr(config, "RETRIEVER", "bm25")
    assert "HE-4330" in rec.buscar(PREGUNTA_004)[0].page_content  # solo BM25: sin la densa equivocada
    with pytest.raises(ValueError):
        rec.buscar("x", modo="inventado")
 
 
def test_el_modo_por_defecto_es_hibrido():
    assert config.RETRIEVER in rec.MODOS
 
 
def test_responder_informa_el_retriever_y_cita_documento_y_pagina(fragmentos, monkeypatch):
    class LLM:
        def invoke(self, prompt):
            return type("R", (), {"content": "ok"})()
    monkeypatch.setattr(rec, "cargar_indice", lambda: IndiceFalso(fragmentos, fragmentos))
    monkeypatch.setattr(rec, "_llm", lambda: LLM())
    monkeypatch.setattr(config, "RETRIEVER", "hibrido")
    monkeypatch.setattr(config, "DESCOMPONER", False)
    r = rec.responder(PREGUNTA_004)
    assert r["retriever"] == "hibrido"
    assert all({"documento", "pagina"} <= set(f) for f in r["fuentes"])
 
 
class LLMSub:
    """LLM falso: devuelve las sub-preguntas que se le indican."""
 
    def __init__(self, preguntas):
        self.preguntas = preguntas
 
    def with_structured_output(self, esquema):
        return self
 
    def invoke(self, prompt):
        return SubPreguntas(preguntas=self.preguntas)
 
 
def test_descomponer_limpia_vacias_y_respeta_el_tope(monkeypatch):
    monkeypatch.setattr(rec, "_llm", lambda: LLMSub(["a", " ", "b", "c", "d"]))
    assert rec.descomponer("x") == ["a", "b", "c"]
 
 
def test_descomponer_sin_resultado_devuelve_la_pregunta_original(monkeypatch):
    monkeypatch.setattr(rec, "_llm", lambda: LLMSub([]))
    assert rec.descomponer("la pregunta") == ["la pregunta"]
 
 
def test_buscar_por_partes_junta_los_temas_sin_repetir(fragmentos, monkeypatch):
    """Cada sub-pregunta trae lo suyo; un fragmento que sale en las dos aparece una vez."""
    uno, dos, tres = fragmentos[0], fragmentos[1], fragmentos[2]
    por_pregunta = {"tema A": [uno, dos], "tema B": [dos, tres]}
    monkeypatch.setattr(rec, "descomponer", lambda p: ["tema A", "tema B"])
    monkeypatch.setattr(rec, "buscar", lambda parte, modo=None: por_pregunta[parte])
    monkeypatch.setattr(config, "DESCOMPONER", True)
    partes, juntos = rec.buscar_por_partes("A y B")
    assert partes == ["tema A", "tema B"]
    assert juntos == [uno, dos, tres]
 
 
def test_sin_descomponer_se_busca_la_pregunta_entera(monkeypatch):
    llamadas = []
    monkeypatch.setattr(rec, "descomponer", lambda p: llamadas.append(p) or ["x"])
    monkeypatch.setattr(rec, "buscar", lambda parte, modo=None: [])
    monkeypatch.setattr(config, "DESCOMPONER", False)
    partes, _ = rec.buscar_por_partes("pregunta entera")
    assert partes == ["pregunta entera"] and llamadas == []
 
 
def test_la_frase_de_sin_informacion_coincide_con_la_que_ignora_la_evaluacion():
    """evaluacion.contiene() no cuenta como acierto esta frase: tiene que ser la del prompt."""
    from mesa_pedidos import evaluacion
    frase = "No tengo informacion suficiente en los documentos para responder eso."
    assert frase in rec.PROMPT
    assert not evaluacion.contiene(frase, ["no"])
 