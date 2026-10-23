"""Pruebas de ejecucion.py: procesar un mensaje, salir en la aprobacion y retomar.
 
LLM simulado, checkpoints en un archivo temporal. No usan API.
"""
import pytest
from test_grafo import simular_llm
 
from mesa_pedidos import ejecucion, nodos
 
PEDIDO_014 = {"SU-3041": 2, "SU-3150": 20}  # Insumos del Litoral: supera su credito
 
 
class Consola:
    """Junta lo que se imprime y contesta lo que se le pregunta."""
 
    def __init__(self, *respuestas):
        self.respuestas = list(respuestas)
        self.lineas = []
 
    def salida(self, texto):
        self.lineas.append(str(texto))
 
    def entrada(self, pregunta):
        return self.respuestas.pop(0)
 
    @property
    def texto(self):
        return "\n".join(self.lineas)
 
 
def test_cargar_mensaje_de_la_bandeja():
    m = ejecucion.cargar_mensaje("MSG-014")
    assert m["cuit"] == "30-71622074-1"
    with pytest.raises(KeyError):
        ejecucion.cargar_mensaje("MSG-999")
 
 
def test_pedido_en_regla_termina_sin_preguntar(monkeypatch, tmp_path):
    simular_llm(monkeypatch, "pedido", {"AB-1020": 200, "AB-1120": 50, "ME-6030": 10})
    c = Consola()
    estado = ejecucion.ejecutar("MSG-009", salida=c.salida, entrada=c.entrada,
                                ruta_checkpoints=tmp_path / "cp.sqlite")
    assert estado == "terminado"
    assert "CONFIRMADO" in c.texto
 
 
def test_el_pedido_frenado_muestra_lo_que_hace_falta_para_decidir(monkeypatch, tmp_path):
    simular_llm(monkeypatch, "pedido", PEDIDO_014)
    c = Consola("s")
    ejecucion.ejecutar("MSG-014", salida=c.salida, entrada=c.entrada,
                       ruta_checkpoints=tmp_path / "cp.sqlite")
    for dato in ("Insumos del Litoral", "30-71622074-1", "$90.000", "limite de credito", "SU-3041"):
        assert dato in c.texto
 
 
def test_salir_y_retomar_el_mismo_pedido(monkeypatch, tmp_path):
    """Corta en la aprobacion (el programa termina), vuelve a correr y retoma."""
    ruta = tmp_path / "cp.sqlite"
    simular_llm(monkeypatch, "pedido", PEDIDO_014)
 
    c1 = Consola("s")  # la persona elige salir y decidir despues
    assert ejecucion.ejecutar("MSG-014", salida=c1.salida, entrada=c1.entrada,
                              ruta_checkpoints=ruta) == "pendiente"
 
    # Otra "terminal": el LLM ya no se llama, el estado sale del archivo
    monkeypatch.setattr(nodos, "_llm", lambda: pytest.fail("No debe volver a clasificar: se retoma"))
    c2 = Consola("a")
    assert ejecucion.ejecutar("MSG-014", salida=c2.salida, entrada=c2.entrada,
                              ruta_checkpoints=ruta) == "terminado"
    assert "retomando" in c2.texto
    assert "CONFIRMADO" in c2.texto
 
 
def test_resolver_rechazar_sin_preguntar(monkeypatch, tmp_path):
    ruta = tmp_path / "cp.sqlite"
    simular_llm(monkeypatch, "pedido", PEDIDO_014)
    c = Consola("s")
    ejecucion.ejecutar("MSG-014", salida=c.salida, entrada=c.entrada, ruta_checkpoints=ruta)
    c2 = Consola()
    ejecucion.ejecutar("MSG-014", resolver="rechazar", motivo="sin credito",
                       salida=c2.salida, entrada=c2.entrada, ruta_checkpoints=ruta)
    assert "RECHAZADO" in c2.texto and "sin credito" in c2.texto
 
 
def test_un_mensaje_ya_procesado_no_se_reprocesa_salvo_reiniciar(monkeypatch, tmp_path):
    ruta = tmp_path / "cp.sqlite"
    simular_llm(monkeypatch, "derivar")
    ejecucion.ejecutar("MSG-018", salida=Consola().salida, ruta_checkpoints=ruta)
    c = Consola()
    ejecucion.ejecutar("MSG-018", salida=c.salida, ruta_checkpoints=ruta)
    assert "ya fue procesado" in c.texto
    c = Consola()
    ejecucion.ejecutar("MSG-018", reiniciar=True, salida=c.salida, ruta_checkpoints=ruta)
    assert "procesando" in c.texto
 