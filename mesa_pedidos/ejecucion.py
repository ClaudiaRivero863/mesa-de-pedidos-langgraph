"""Procesa un mensaje de la bandeja y maneja el pedido de aprobacion.
 
La idea clave: el grafo guarda su estado en un ARCHIVO (checkpoints.sqlite), con un
thread_id por mensaje. Si un pedido queda esperando aprobacion y el programa termina,
volver a correr el mismo mensaje lo RETOMA donde quedo en vez de empezar de cero.
"""
import json
 
from langgraph.types import Command
 
from . import config
from .grafo import abrir_checkpointer, construir_grafo
 
 
def _pesos(x) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")
 
 
def cargar_mensaje(mensaje_id: str) -> dict:
    with open(config.DIR_DATOS / "bandeja.jsonl", encoding="utf-8") as f:
        for linea in f:
            mensaje = json.loads(linea)
            if mensaje["id"] == mensaje_id:
                return mensaje
    raise KeyError(f"No hay ningun mensaje {mensaje_id} en la bandeja.")
 
 
def formatear_payload(p: dict) -> str:
    """Lo que ve la persona que decide: todo en una pantalla."""
    lineas = [
        "",
        "=" * 64,
        f"PEDIDO PENDIENTE DE APROBACION  ({p['mensaje_id']})",
        "=" * 64,
        f"Cliente:  {p['razon_social']}  (CUIT {p['cuit']})",
        f"Credito:  limite {_pesos(p['limite_credito'])} | saldo {_pesos(p['saldo_actual'])}"
        f" | disponible {_pesos(p['credito_disponible'])}",
        f"Mora:     {p['mora_dias']} dias",
        f"Neto del pedido: {_pesos(p['neto'])}",
        "",
        "Alertas:",
    ]
    lineas += [f"  ! {a}" for a in p["alertas"]]
    lineas += ["", "Renglones:"]
    for r in p["renglones"]:
        lineas.append(
            f"  {r['sku']}  x{r['cantidad']:<5} {_pesos(r['subtotal']):>14}"
            f"  (dto {r['descuento_pct']:g}%, stock {r['stock']}, dep. {r['deposito']})"
        )
    lineas.append("=" * 64)
    return "\n".join(lineas)
 
 
def mostrar_resultado(estado: dict, salida=print) -> None:
    salida(f"\nResultado: {estado.get('estado_final', '?').upper()}")
    if estado.get("intencion"):
        salida(f"Clasificado como: {estado['intencion']}")
    if estado.get("respuesta"):
        salida(f"\n{estado['respuesta']}")
        salida("Fuentes consultadas: " + ", ".join(
            f"{f['documento']} p.{f['pagina']}" for f in estado.get("fuentes", [])))
    if estado.get("neto") is not None and estado.get("renglones"):
        salida(f"Neto: {_pesos(estado['neto'])}")
    for alerta in estado.get("alertas", []):
        salida(f"  ! {alerta}")
    salida("\nRecorrido (log):")
    for linea in estado.get("log", []):
        salida(f"  - {linea}")
 
 
def pedir_decision(entrada=input) -> dict:
    """Pregunta a la persona. Devuelve {"accion": "aprobar"|"rechazar"|"salir", ...}."""
    while True:
        r = entrada("Decision: [a]probar, [r]echazar o [s]alir y decidir despues? ").strip().lower()
        if r in ("a", "aprobar"):
            return {"accion": "aprobar"}
        if r in ("r", "rechazar"):
            return {"accion": "rechazar", "motivo": entrada("Motivo (opcional): ").strip()}
        if r in ("s", "salir"):
            return {"accion": "salir"}
        print("Opcion no valida.")
 
 
def ejecutar(mensaje_id, resolver=None, motivo="", reiniciar=False,
             entrada=input, salida=print, ruta_checkpoints=None) -> str:
    """Devuelve "terminado" o "pendiente" (esperando una decision humana)."""
    mensaje = cargar_mensaje(mensaje_id)
    checkpointer = abrir_checkpointer(ruta_checkpoints)
    try:
        grafo = construir_grafo(checkpointer)
        cfg = {"configurable": {"thread_id": mensaje_id}}  # un thread_id por mensaje
 
        if reiniciar:
            checkpointer.delete_thread(mensaje_id)
        snapshot = grafo.get_state(cfg)
 
        if snapshot.next:  # quedo frenado en una corrida anterior
            salida(f"{mensaje_id}: retomando el pedido que quedo esperando aprobacion.")
            payload = snapshot.tasks[0].interrupts[0].value
        elif snapshot.values.get("estado_final"):
            salida(f"{mensaje_id}: ya fue procesado. Usa --reiniciar para volver a correrlo.")
            mostrar_resultado(snapshot.values, salida)
            return "terminado"
        else:
            salida(f"{mensaje_id}: procesando...  ({mensaje['texto'][:70]})")
            resultado = grafo.invoke(
                {"mensaje_id": mensaje_id, "texto": mensaje["texto"], "cuit": mensaje["cuit"],
                 "fecha": mensaje.get("fecha")}, cfg)
            if "__interrupt__" not in resultado:
                mostrar_resultado(resultado, salida)
                return "terminado"
            payload = resultado["__interrupt__"][0].value
 
        salida(formatear_payload(payload))
        decision = ({"accion": resolver, "motivo": motivo} if resolver else pedir_decision(entrada))
        if decision["accion"] == "salir":
            salida(f"\nQueda pendiente. Para retomarlo: python correr.py {mensaje_id}")
            return "pendiente"
        resultado = grafo.invoke(Command(resume=decision), cfg)
        mostrar_resultado(resultado, salida)
        return "terminado"
    finally:
        checkpointer.conn.close()
 