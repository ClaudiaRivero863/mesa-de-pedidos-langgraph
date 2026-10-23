"""El grafo: nueve nodos, tres puntos de ruteo, cinco hojas.
 
Los routers viven aca, declarados en el grafo, y no como `if` dentro de un nodo.
Un router NO decide nada de negocio: traduce una decision que un nodo ya escribio
en el estado (intencion, decision, aprobacion) al nombre de una arista.
"""
import sqlite3
from pathlib import Path
 
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
 
from . import config, nodos
from .estado import Estado
 
 
# --------------------------------------------------------------------- routers
def ruta_por_intencion(estado: Estado) -> str:
    """Despues de clasificar: consulta | pedido | derivar."""
    return estado["intencion"]
 
 
def ruta_por_validacion(estado: Estado) -> str:
    """Despues de validar el pedido: confirmar | humano | rechazar | derivar."""
    return estado["decision"]
 
 
def ruta_post_aprobacion(estado: Estado) -> str:
    """Despues del interrupt: aprueba | rechaza."""
    return "aprueba" if estado["aprobacion"] == "aprobado" else "rechaza"
 
 
# ----------------------------------------------------------------------- grafo
def construir_grafo(checkpointer=None):
    g = StateGraph(Estado)
 
    g.add_node("cargar_cliente", nodos.cargar_cliente)
    g.add_node("clasificar", nodos.clasificar)
    g.add_node("responder_consulta", nodos.responder_consulta)
    g.add_node("extraer_pedido", nodos.extraer_pedido)
    g.add_node("validar_pedido", nodos.validar_pedido)
    g.add_node("aprobacion_humana", nodos.aprobacion_humana)
    g.add_node("confirmar_pedido", nodos.confirmar_pedido)
    g.add_node("rechazar_pedido", nodos.rechazar_pedido)
    g.add_node("derivar_humano", nodos.derivar_humano)
 
    g.add_edge(START, "cargar_cliente")
    g.add_edge("cargar_cliente", "clasificar")
 
    g.add_conditional_edges("clasificar", ruta_por_intencion, {
        "consulta": "responder_consulta",
        "pedido": "extraer_pedido",
        "derivar": "derivar_humano",
    })
    g.add_edge("extraer_pedido", "validar_pedido")
 
    g.add_conditional_edges("validar_pedido", ruta_por_validacion, {
        "confirmar": "confirmar_pedido",
        "humano": "aprobacion_humana",
        "rechazar": "rechazar_pedido",
        "derivar": "derivar_humano",
    })
 
    g.add_conditional_edges("aprobacion_humana", ruta_post_aprobacion, {
        "aprueba": "confirmar_pedido",
        "rechaza": "rechazar_pedido",
    })
 
    for hoja in ("responder_consulta", "confirmar_pedido", "rechazar_pedido", "derivar_humano"):
        g.add_edge(hoja, END)
 
    return g.compile(checkpointer=checkpointer)
 
 
def abrir_checkpointer(ruta=None) -> SqliteSaver:
    """Checkpointer sobre ARCHIVO: el estado sobrevive al cierre del programa.
 
    Un thread_id por mensaje (el id del mensaje) indexa cada historia.
    """
    ruta = ruta or config.DIR_ARTEFACTOS / "checkpoints.sqlite"
    if str(ruta) != ":memory:":  # ":memory:" es una base temporal, solo para pruebas
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(str(ruta), check_same_thread=False)
    return SqliteSaver(conexion)