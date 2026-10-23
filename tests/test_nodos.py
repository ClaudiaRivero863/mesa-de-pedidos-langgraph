"""Pruebas de nodos.py. El LLM y el RAG se reemplazan por simulados: no usan API."""
from mesa_pedidos import nodos
from mesa_pedidos.esquemas import Clasificacion, ItemPedido, PedidoExtraido
 
 
class LLMFalso:
    """Imita ChatOpenAI().with_structured_output(Esquema).invoke(prompt)."""
 
    def __init__(self, respuesta):
        self.respuesta = respuesta
 
    def with_structured_output(self, esquema):
        return self
 
    def invoke(self, prompt):
        return self.respuesta
 
 
def test_cargar_cliente_usa_los_datos_reales():
    r = nodos.cargar_cliente({"cuit": "30-70988341-7"})
    assert r["cliente"]["razon_social"] == "Metalurgica San Jorge S.A."
    assert r["log"][0].startswith("cargar_cliente")
 
 
def test_cargar_cliente_con_cuit_inexistente_no_falla():
    r = nodos.cargar_cliente({"cuit": "00-00000000-0"})
    assert r["cliente"]["encontrado"] is False
 
 
def test_clasificar_guarda_intencion_y_motivo(monkeypatch):
    falso = LLMFalso(Clasificacion(intencion="consulta", motivo="pregunta por flete"))
    monkeypatch.setattr(nodos, "_llm", lambda: falso)
    r = nodos.clasificar({"texto": "cuanto sale el flete"})
    assert r["intencion"] == "consulta"
    assert r["motivo_clasificacion"] == "pregunta por flete"
 
 
def test_extraer_pedido_normaliza_sku_y_conserva_extras(monkeypatch):
    pedido = PedidoExtraido(
        items=[ItemPedido(sku=" ab-1020 ", cantidad=200)],
        descuento_solicitado=25, pago_anticipado=True,
    )
    monkeypatch.setattr(nodos, "_llm", lambda: LLMFalso(pedido))
    r = nodos.extraer_pedido({"texto": "..."})
    assert r["items"] == [{"sku": "AB-1020", "cantidad": 200}]
    assert r["descuento_solicitado"] == 25
    assert r["pago_anticipado"] is True
 
 
def test_responder_consulta_guarda_respuesta_y_fuentes(monkeypatch):
    falso = {"respuesta": "175.000", "fuentes": [{"documento": "logistica_fletes_2026.pdf", "pagina": 1}]}
    monkeypatch.setattr(nodos, "responder", lambda texto, fecha=None: falso)
    r = nodos.responder_consulta({"texto": "flete a Neuquen"})
    assert r["respuesta"] == "175.000"
    assert r["estado_final"] == "respondido"
 
 
def estado_de_pedido(cuit, items, **extra):
    cliente = nodos.cargar_cliente({"cuit": cuit})["cliente"]
    return {"cliente": cliente, "items": items, **extra}
 
 
def test_validar_pedido_confirma_un_pedido_en_regla():
    est = estado_de_pedido("30-71455982-2", [
        {"sku": "AB-1020", "cantidad": 200}, {"sku": "AB-1120", "cantidad": 50},
        {"sku": "ME-6030", "cantidad": 10}])
    r = nodos.validar_pedido(est)
    assert r["decision"] == "confirmar"
    assert len(r["renglones"]) == 3
    assert r["neto"] > 0
 
 
def test_validar_pedido_manda_a_una_persona_si_supera_el_credito():
    est = estado_de_pedido("30-71622074-1", [{"sku": "SU-3041", "cantidad": 2}])
    r = nodos.validar_pedido(est)
    assert r["decision"] == "humano"
    assert any("limite de credito" in a for a in r["alertas"])
 
 
def test_validar_pedido_sin_items_deriva():
    r = nodos.validar_pedido(estado_de_pedido("30-71455982-2", []))
    assert r["decision"] == "derivar"
 
 
def test_nodos_finales_marcan_el_estado_y_agregan_log():
    assert nodos.confirmar_pedido({})["estado_final"] == "confirmado"
    assert nodos.rechazar_pedido({})["estado_final"] == "rechazado"
    r = nodos.derivar_humano({"motivo_clasificacion": "es un reclamo"})
    assert r["estado_final"] == "derivado"
    assert "es un reclamo" in r["log"][0]
 