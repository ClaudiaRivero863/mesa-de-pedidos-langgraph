"""Los nodos del grafo.
 
Cada nodo recibe el estado y devuelve SOLO los campos que cambio, siempre con una
linea para `log`. No llaman al nodo siguiente: el orden lo define grafo.py.
"""
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
 
from . import config
from .esquemas import Clasificacion, PedidoExtraido
from .estado import Estado
from .herramientas import consultar_producto, verificar_credito
from .prompts import PROMPT_CLASIFICAR, PROMPT_EXTRAER
from .recuperacion import responder
from .reglas import evaluar_pedido
 
 
def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.MODELO_LLM, temperature=0)
 
 
def cargar_cliente(estado: Estado) -> dict:
    """Resuelve el CUIT contra clientes.csv (tool, sin LLM)."""
    cliente = verificar_credito.invoke({"cuit": estado["cuit"], "monto_neto": 0})
    nombre = cliente.get("razon_social", f"CUIT {estado['cuit']} no encontrado")
    return {"cliente": cliente, "log": [f"cargar_cliente: {nombre}"]}
 
 
def clasificar(estado: Estado) -> dict:
    """LLM + structured output: consulta, pedido o derivar."""
    prompt = PROMPT_CLASIFICAR.format_messages(texto=estado["texto"])
    r = _llm().with_structured_output(Clasificacion).invoke(prompt)
    return {
        "intencion": r.intencion,
        "motivo_clasificacion": r.motivo,
        "log": [f"clasificar: {r.intencion} ({r.motivo})"],
    }
 
 
def responder_consulta(estado: Estado) -> dict:
    """RAG: busca en el indice y responde citando documento y pagina."""
    res = responder(estado["texto"], fecha=estado.get("fecha"))
    return {
        "respuesta": res["respuesta"],
        "fuentes": res["fuentes"],
        "estado_final": "respondido",
        "log": [f"responder_consulta: {len(res['fuentes'])} fragmentos consultados"],
    }
 
 
def extraer_pedido(estado: Estado) -> dict:
    """LLM + structured output: items, descuento pedido y pago anticipado."""
    prompt = PROMPT_EXTRAER.format_messages(texto=estado["texto"])
    r = _llm().with_structured_output(PedidoExtraido).invoke(prompt)
    items = [{"sku": i.sku.strip().upper(), "cantidad": i.cantidad} for i in r.items]
    return {
        "items": items,
        "descuento_solicitado": r.descuento_solicitado,
        "pago_anticipado": r.pago_anticipado,
        "log": [f"extraer_pedido: {len(items)} items"],
    }
 
 
def validar_pedido(estado: Estado) -> dict:
    """Tools + reglas, SIN LLM: consulta cada producto y decide con reglas.py."""
    items = estado.get("items", [])
    productos = {i["sku"]: consultar_producto.invoke({"sku": i["sku"]}) for i in items}
    r = evaluar_pedido(
        estado.get("cliente", {}),
        items,
        productos,
        descuento_solicitado=estado.get("descuento_solicitado"),
        pago_anticipado=estado.get("pago_anticipado", False),
    )
    return {
        "renglones": r["renglones"],
        "neto": r["neto"],
        "alertas": r["alertas"],
        "decision": r["decision"],
        "log": [f"validar_pedido: {r['decision']} (neto ${r['neto']:,.0f}, {len(r['alertas'])} alertas)"],
    }
 
 
def aprobacion_humana(estado: Estado) -> dict:
    """Human-in-the-loop: frena el grafo con interrupt() y espera a una persona.
 
    El payload trae todo lo que hace falta para decidir sin abrir otra pantalla.
    Se retoma con Command(resume={"accion": "aprobar" | "rechazar", "motivo": "..."}).
    Ojo: al retomar, LangGraph vuelve a ejecutar este nodo desde el principio,
    por eso lo que hay antes del interrupt no debe tener efectos.
    """
    cliente = estado.get("cliente", {})
    respuesta = interrupt({
        "mensaje_id": estado.get("mensaje_id"),
        "razon_social": cliente.get("razon_social"),
        "cuit": cliente.get("cuit"),
        "limite_credito": cliente.get("limite_credito"),
        "saldo_actual": cliente.get("saldo_actual"),
        "credito_disponible": cliente.get("credito_disponible"),
        "mora_dias": cliente.get("mora_dias"),
        "neto": estado.get("neto"),
        "alertas": estado.get("alertas", []),
        "renglones": estado.get("renglones", []),
        "opciones": 'Responder {"accion": "aprobar"} o {"accion": "rechazar", "motivo": "..."}',
    })
    accion = respuesta.get("accion") if isinstance(respuesta, dict) else respuesta
    motivo = respuesta.get("motivo", "") if isinstance(respuesta, dict) else ""
    aprobado = str(accion).strip().lower() == "aprobar"
    return {
        "aprobacion": "aprobado" if aprobado else "rechazado",
        "log": [f"aprobacion_humana: {'aprobado' if aprobado else 'rechazado'}" + (f" ({motivo})" if motivo else "")],
    }
 
 
def confirmar_pedido(estado: Estado) -> dict:
    return {"estado_final": "confirmado", "log": ["confirmar_pedido: pedido confirmado"]}
 
 
def rechazar_pedido(estado: Estado) -> dict:
    return {"estado_final": "rechazado", "log": ["rechazar_pedido: pedido rechazado"]}
 
 
def derivar_humano(estado: Estado) -> dict:
    motivo = estado.get("motivo_clasificacion") or "; ".join(estado.get("alertas", []))
    return {"estado_final": "derivado", "log": [f"derivar_humano: {motivo}"]}
 