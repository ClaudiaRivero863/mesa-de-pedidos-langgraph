"""Pruebas de la medicion. LLM y RAG simulados: no usan API."""
import json
 
from test_grafo import simular_llm
 
from mesa_pedidos import config, evaluacion, nodos
from mesa_pedidos.esquemas import Clasificacion
 
 
def test_ruta_de_un_pedido_frenado_es_aprobacion_humana():
    assert evaluacion.ruta_de({"log": ["validar_pedido: humano"]}, pendiente=True) == "aprobacion_humana"
 
 
def test_ruta_de_sale_del_ultimo_nodo_del_log():
    log = ["cargar_cliente: x", "clasificar: consulta", "responder_consulta: 4 fragmentos"]
    assert evaluacion.ruta_de({"log": log}, pendiente=False) == "responder_consulta"
    assert evaluacion.ruta_de({"log": ["confirmar_pedido: ok"]}, pendiente=False) == "confirmar_pedido"
 
 
def test_contiene_no_distingue_mayusculas_y_pide_todos_los_terminos():
    assert evaluacion.contiene("Descuento del 5% sobre 1.500.000", ["5%", "1.500.000"])
    assert evaluacion.contiene("Bateria HE-4335", ["he-4335"])
    assert not evaluacion.contiene("Descuento del 5%", ["5%", "1.500.000"])
 
 
def test_contiene_no_acepta_numeros_dentro_de_otros_ni_tildes_distintas():
    assert not evaluacion.contiene("Vale 15 dias", ["5"])
    assert evaluacion.contiene("Vale 5 dias habiles", ["5"])
    assert evaluacion.contiene("La batería es la HE-4335.", ["bateria"])
    assert evaluacion.contiene("Flete de $175.000.", ["175.000"])
 
 
def test_contiene_no_cuenta_como_acierto_el_no_tengo_informacion():
    """'no' esta dentro de 'No tengo informacion': una no-respuesta no puede pasar por dato clave."""
    assert not evaluacion.contiene("No tengo informacion suficiente en los documentos para responder eso.", ["no"])
    assert evaluacion.contiene("No se aceptan devoluciones de zapatos de seguridad.", ["no"])
 
 
def test_resumir_cuenta_cada_numero():
    filas = [
        {"ruta_ok": True, "dato_clave_ok": True, "documento_ok": True},
        {"ruta_ok": True, "dato_clave_ok": False, "documento_ok": True},
        {"ruta_ok": False, "items_ok": True},
        {"ruta_ok": True, "items_ok": False},
    ]
    r = evaluacion.resumir(filas)
    assert (r["ramas_correctas"], r["total"]) == (3, 4)
    assert (r["dato_clave"], r["consultas"]) == (1, 2)
    assert (r["items_exactos"], r["pedidos"]) == (1, 2)
 
 
def test_corrida_completa_con_resultados_conocidos(monkeypatch, tmp_path):
    """Si todo se clasifica como consulta, solo las 8 consultas salen por la rama correcta."""
    simular_llm(monkeypatch, "consulta")
    monkeypatch.setattr(nodos, "responder", lambda t, fecha=None: {
        "respuesta": "5% 1.500.000 175.000 8 no HE-4335 12 meses 24", "fuentes": []})
    monkeypatch.setattr(config, "DIR_ARTEFACTOS", tmp_path)
 
    lineas = []
    r = evaluacion.evaluar(etiqueta="prueba", salida=lineas.append)
 
    assert r["total"] == 20 and r["consultas"] == 8 and r["pedidos"] == 9
    assert r["ramas_correctas"] == 8
    assert r["dato_clave"] == 8
    guardado = json.loads((tmp_path / "evaluacion_prueba.json").read_text(encoding="utf-8"))
    assert guardado["metricas"]["ramas_correctas"] == 8
    assert len(guardado["casos"]) == 20
    assert any("RAMA CORRECTA : 8/20" in linea for linea in lineas)
 
 
def test_un_error_en_un_caso_no_tira_abajo_la_corrida(monkeypatch):
    def romper(texto, fecha=None):
        raise RuntimeError("fallo de red")
    simular_llm(monkeypatch, "consulta")
    monkeypatch.setattr(nodos, "responder", romper)
    fila = evaluacion.evaluar_caso(
        {"id": "MSG-001", "texto": "x", "cuit": "30-71455982-2"},
        {"id": "MSG-001", "intencion": "consulta", "ruta": "responder_consulta",
         "documento": "d.pdf", "debe_contener": ["5%"]})
    assert fila["ruta"] == "ERROR" and fila["ruta_ok"] is False
    assert "fallo de red" in fila["error"]
 