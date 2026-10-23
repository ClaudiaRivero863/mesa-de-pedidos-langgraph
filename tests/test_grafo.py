"""Pruebas del grafo completo. LLM y RAG simulados: no usan API.
 
Recorren cada rama y, sobre todo, el frenado: cortar a mitad de una aprobacion,
cerrar la conexion, abrir otra y retomar el MISMO pedido.
"""
import pytest
from langgraph.types import Command
 
from mesa_pedidos import grafo, nodos
from mesa_pedidos.esquemas import Clasificacion, ItemPedido, PedidoExtraido
 
 
class LLMFalso:
    """Devuelve una respuesta distinta segun el esquema que se le pida."""
 
    def __init__(self, respuestas):
        self.respuestas = respuestas
        self._esquema = None
 
    def with_structured_output(self, esquema):
        copia = LLMFalso(self.respuestas)
        copia._esquema = esquema
        return copia
 
    def invoke(self, prompt):
        return self.respuestas[self._esquema]
 
 
def simular_llm(monkeypatch, intencion, items=None, **extra):
    respuestas = {Clasificacion: Clasificacion(intencion=intencion, motivo="simulado")}
    if items is not None:
        respuestas[PedidoExtraido] = PedidoExtraido(
            items=[ItemPedido(sku=s, cantidad=c) for s, c in items.items()], **extra)
    monkeypatch.setattr(nodos, "_llm", lambda: LLMFalso(respuestas))
 
 
def correr(cuit, texto="mensaje", mensaje_id="MSG-TEST", checkpointer=None):
    g = grafo.construir_grafo(checkpointer)
    config = {"configurable": {"thread_id": mensaje_id}}
    entrada = {"mensaje_id": mensaje_id, "texto": texto, "cuit": cuit}
    return g, config, g.invoke(entrada, config)
 
 
def test_el_grafo_compila_con_los_nueve_nodos():
    mermaid = grafo.construir_grafo().get_graph().draw_mermaid()
    for nodo in ("cargar_cliente", "clasificar", "responder_consulta", "extraer_pedido",
                 "validar_pedido", "aprobacion_humana", "confirmar_pedido",
                 "rechazar_pedido", "derivar_humano"):
        assert nodo in mermaid
 
 
def test_rama_consulta(monkeypatch):
    simular_llm(monkeypatch, "consulta")
    monkeypatch.setattr(nodos, "responder", lambda t, fecha=None: {"respuesta": "175.000", "fuentes": []})
    _, _, r = correr("30-69874512-2")
    assert r["estado_final"] == "respondido"
    assert len(r["log"]) == 3  # cargar_cliente, clasificar, responder_consulta
 
 
def test_rama_derivar(monkeypatch):
    simular_llm(monkeypatch, "derivar")
    _, _, r = correr("30-71455982-2")
    assert r["estado_final"] == "derivado"
 
 
def test_rama_confirmar_un_pedido_en_regla(monkeypatch):
    simular_llm(monkeypatch, "pedido", {"AB-1020": 200, "AB-1120": 50, "ME-6030": 10})
    _, _, r = correr("30-71455982-2")
    assert r["estado_final"] == "confirmado"
    assert len(r["log"]) == 5  # un renglon por nodo: el reducer concatena, no pisa
 
 
def test_rama_rechazar_cliente_suspendido(monkeypatch):
    simular_llm(monkeypatch, "pedido", {"SE-5090": 10, "SU-3200": 20})
    _, _, r = correr("30-71788450-3")
    assert r["estado_final"] == "rechazado"
 
 
def test_pedido_dudoso_frena_en_aprobacion_humana(monkeypatch):
    simular_llm(monkeypatch, "pedido", {"SU-3041": 2, "SU-3150": 20})
    g, config, r = correr("30-71622074-1", checkpointer=grafo.abrir_checkpointer(":memory:"))
    assert "__interrupt__" in r
    payload = r["__interrupt__"][0].value
    assert payload["razon_social"] == "Insumos del Litoral S.R.L."
    assert payload["credito_disponible"] == 90000.0
    assert payload["neto"] > 0 and payload["alertas"] and payload["renglones"]
    assert g.get_state(config).next == ("aprobacion_humana",)
 
 
@pytest.mark.parametrize("accion,final", [("aprobar", "confirmado"), ("rechazar", "rechazado")])
def test_retomar_con_aprobar_o_rechazar(monkeypatch, accion, final):
    simular_llm(monkeypatch, "pedido", {"SU-3041": 2, "SU-3150": 20})
    g, config, _ = correr("30-71622074-1", checkpointer=grafo.abrir_checkpointer(":memory:"))
    r = g.invoke(Command(resume={"accion": accion}), config)
    assert r["estado_final"] == final
    assert "aprobacion_humana" in " ".join(r["log"])
 
 
def test_retomar_despues_de_cerrar_la_conexion(monkeypatch, tmp_path):
    """El corazon del requisito: cortar, cerrar, volver a abrir y retomar el mismo pedido."""
    archivo = tmp_path / "checkpoints.sqlite"
    simular_llm(monkeypatch, "pedido", {"SU-3041": 2, "SU-3150": 20})
 
    primero = grafo.abrir_checkpointer(archivo)
    _, config, r = correr("30-71622074-1", mensaje_id="MSG-014", checkpointer=primero)
    assert "__interrupt__" in r
    primero.conn.close()  # el "programa" termina
 
    segundo = grafo.abrir_checkpointer(archivo)  # otra conexion, como una terminal nueva
    g2 = grafo.construir_grafo(segundo)
    assert g2.get_state(config).next == ("aprobacion_humana",)
    r = g2.invoke(Command(resume={"accion": "aprobar"}), config)
    assert r["estado_final"] == "confirmado"
    assert r["mensaje_id"] == "MSG-014"
 