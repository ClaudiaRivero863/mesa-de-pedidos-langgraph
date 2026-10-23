"""La interfaz de Streamlit abre y recorre las ramas. LLM simulado: no usa API."""
import pytest
 
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
 
from test_grafo import simular_llm  # noqa: E402
 
from mesa_pedidos import config, nodos  # noqa: E402
 
APP = str(config.RAIZ / "app.py")
 
 
def abrir(monkeypatch, tmp_path, intencion, items=None, **extra):
    simular_llm(monkeypatch, intencion, items, **extra)
    monkeypatch.setattr(nodos, "responder", lambda t, fecha=None: {
        "respuesta": "El flete es $175.000.",
        "fuentes": [{"documento": "logistica_fletes_2026.pdf", "pagina": 1, "seccion": "2. ZONAS"}]})
    monkeypatch.setattr(config, "DIR_ARTEFACTOS", tmp_path)
    monkeypatch.setattr(config, "DIR_INDICE", tmp_path)  # que no marque "falta el indice"
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at
 
 
def texto_pantalla(at) -> str:
    return " ".join(m.value for m in at.markdown)
 
 
def metrica(at, titulo):
    return next(m for m in at.metric if m.label == titulo).value
 
 
def procesar_mensaje(at, mensaje_id):
    radio = next(r for r in at.radio if r.label == "Mensajes")
    radio.set_value(next(o for o in radio.options if mensaje_id in str(o))).run()
    next(b for b in at.button if b.label == "Procesar este mensaje").click().run()
    assert not at.exception
    return at
 
 
def test_la_pantalla_abre_con_la_cabecera_y_los_indicadores(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "consulta")
    assert "Mesa de Pedidos" in texto_pantalla(at)
    assert metrica(at, "Esperan tu aprobación") == "0"
    assert len(at.tabs) == 0  # sin panel tecnico por defecto
 
 
def test_el_panel_tecnico_aparece_con_el_interruptor(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "consulta")
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "🔧 Panel técnico" in [t.label for t in at.tabs]
 
 
def test_una_consulta_muestra_la_respuesta_la_etiqueta_y_cuenta_como_respondida(monkeypatch, tmp_path):
    at = procesar_mensaje(abrir(monkeypatch, tmp_path, "consulta"), "MSG-002")
    pantalla = texto_pantalla(at)
    assert "$175.000" in pantalla and "Respondido" in pantalla
    assert metrica(at, "Respondidos") == "1"
 
 
def test_el_detalle_tecnico_dibuja_el_grafo_con_el_camino_resaltado(monkeypatch, tmp_path):
    at = procesar_mensaje(abrir(monkeypatch, tmp_path, "consulta"), "MSG-002")
    dot = str(at.get("graphviz_chart")[0].proto)
    assert "responder_consulta" in dot and "#BBF7D0" in dot  # verde: lo que recorrio
 
 
def test_un_pedido_que_pide_aprobacion_espera_y_se_puede_rechazar(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "pedido", {"HE-4180": 500}, descuento_solicitado=50)
    procesar_mensaje(at, "MSG-013")
    assert "Espera aprobación" in texto_pantalla(at)
    assert metrica(at, "Esperan tu aprobación") == "1"
    next(b for b in at.button if "Rechazar" in b.label).click().run()
    assert not at.exception
    assert "Rechazado por decisión humana" in texto_pantalla(at)
    assert metrica(at, "Rechazados") == "1" and metrica(at, "Esperan tu aprobación") == "0"
 
 
def test_un_pedido_aprobado_cuenta_como_confirmado(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "pedido", {"HE-4180": 500}, descuento_solicitado=50)
    procesar_mensaje(at, "MSG-013")
    next(b for b in at.button if "Aprobar" in b.label).click().run()
    assert not at.exception
    assert "Aprobado y confirmado" in texto_pantalla(at)
    assert metrica(at, "Pedidos confirmados") == "1"
 
 
def test_procesar_toda_la_bandeja_y_filtrar_por_estado(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "derivar")
    next(b for b in at.button if "Procesar los 20" in b.label).click().run()
    assert not at.exception
    assert metrica(at, "Derivados a una persona") == "20"
    filtro = next(r for r in at.radio if r.label == "Mostrar")
    filtro.set_value("pendientes").run()
    assert any("No hay mensajes en esta categoría" in i.value for i in at.info)
    next(r for r in at.radio if r.label == "Mostrar").set_value("derivados").run()
    assert len(next(r for r in at.radio if r.label == "Mensajes").options) == 20
 
 
def test_al_abrir_recupera_lo_que_ya_estaba_procesado(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "pedido", {"HE-4180": 500}, descuento_solicitado=50)
    procesar_mensaje(at, "MSG-013")  # queda esperando aprobacion en el archivo de checkpoints
    otra_sesion = AppTest.from_file(APP, default_timeout=60)
    otra_sesion.run()
    assert not otra_sesion.exception
    assert metrica(otra_sesion, "Esperan tu aprobación") == "1"
 
 
def test_un_intento_de_manipulacion_se_prueba_y_no_se_confirma(monkeypatch, tmp_path):
    at = abrir(monkeypatch, tmp_path, "pedido", {"HE-4180": 500}, descuento_solicitado=50)
    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "Ejemplos").select("Intento de manipulación").run()
    next(b for b in at.button if b.label == "Probar mensaje").click().run()
    assert not at.exception
    assert any("no se confirmó" in s.value for s in at.success)
 