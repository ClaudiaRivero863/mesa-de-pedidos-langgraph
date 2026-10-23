"""El estado del grafo: lo unico que viaja de nodo en nodo.
 
Cada nodo recibe este diccionario y devuelve SOLO los campos que cambio.
total=False porque al principio la mayoria de los campos todavia no existe:
cada nodo va completando los suyos.
"""
from operator import add
from typing import Annotated, TypedDict
 
 
class Estado(TypedDict, total=False):
    # --- Entrada (la completa quien llama al grafo)
    mensaje_id: str          # ej. "MSG-009"; tambien se usa como thread_id
    texto: str               # el mensaje del cliente
    cuit: str                # CUIT que viene en la bandeja
    fecha: str               # fecha del mensaje (AAAA-MM-DD), para plazos y vencimientos
 
    # --- cargar_cliente
    cliente: dict            # fila del cliente (resultado de verificar_credito)
 
    # --- clasificar
    intencion: str           # "consulta" | "pedido" | "derivar"
    motivo_clasificacion: str
 
    # --- responder_consulta (RAG)
    respuesta: str
    fuentes: list[dict]      # [{"documento": ..., "pagina": ...}]
 
    # --- extraer_pedido
    items: list[dict]        # [{"sku": ..., "cantidad": ...}]
    descuento_solicitado: float   # % pedido por el cliente, si pidio alguno
    pago_anticipado: bool         # el cliente dice que paga antes del envio
 
    # --- validar_pedido (tools + reglas, sin LLM)
    renglones: list[dict]    # items con precio, subtotal y stock
    neto: float              # monto neto del pedido, sin IVA
    alertas: list[str]       # motivos por los que hace falta una persona
    decision: str            # "confirmar" | "humano" | "rechazar" | "derivar"
 
    # --- aprobacion_humana
    aprobacion: str          # "aprobado" | "rechazado"
 
    # --- resultado final
    estado_final: str        # "confirmado" | "rechazado" | "derivado" | "respondido"
 
    # --- auditoria: con reducer `add`, cada nodo AGREGA su linea en vez de pisar la lista
    log: Annotated[list[str], add]
 