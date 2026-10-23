"""Mide el sistema con los 20 mensajes de la bandeja contra el golden set.
 
Dos numeros obligatorios:
  1. cuantos mensajes salieron por la rama correcta (de 20)
  2. en cuantas de las 8 consultas aparece el dato clave (debe_contener)
Y dos informativos, utiles para entender un fallo:
  - si la consulta cito el documento correcto
  - si la extraccion de items fue exacta (de los 9 pedidos)
"""
import json
import re
import unicodedata
 
from . import config
from .grafo import abrir_checkpointer, construir_grafo
 
 
def ruta_de(estado: dict, pendiente: bool) -> str:
    """Hoja a la que llego el mensaje. Un pedido frenado se queda en aprobacion_humana."""
    if pendiente:
        return "aprobacion_humana"
    log = estado.get("log") or ["?"]
    return log[-1].split(":")[0]  # cada nodo escribe "nombre_del_nodo: ..."
 
 
def _normalizar(texto: str) -> str:
    """Minusculas y sin tildes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
 
 
def contiene(respuesta: str, terminos: list[str]) -> bool:
    """Todos los terminos aparecen en la respuesta como palabras o cifras completas.
 
    Sin distinguir mayusculas ni tildes. Un '5' no se encuentra dentro de '15', y una
    respuesta que dice 'No tengo informacion suficiente' no cuenta como acierto aunque
    contenga la palabra 'no'.
    """
    texto = _normalizar(respuesta)
    if "no tengo informacion suficiente" in texto:
        return False
    return all(
        re.search(rf"(?<!\w){re.escape(_normalizar(t))}(?!\w)", texto) for t in terminos
    )
 
 
def correr_caso(mensaje: dict) -> tuple[dict, bool]:
    """Procesa un mensaje con un checkpointer en memoria: no toca los checkpoints reales."""
    checkpointer = abrir_checkpointer(":memory:")
    try:
        grafo = construir_grafo(checkpointer)
        cfg = {"configurable": {"thread_id": mensaje["id"]}}
        grafo.invoke({"mensaje_id": mensaje["id"], "texto": mensaje["texto"], "cuit": mensaje["cuit"],
                      "fecha": mensaje.get("fecha")}, cfg)
        snapshot = grafo.get_state(cfg)
        return dict(snapshot.values), bool(snapshot.next)
    finally:
        checkpointer.conn.close()
 
 
def evaluar_caso(mensaje: dict, caso: dict) -> dict:
    try:
        estado, pendiente = correr_caso(mensaje)
        error = None
    except Exception as e:  # un fallo en un caso no debe tirar abajo los otros 19
        estado, pendiente, error = {}, False, f"{type(e).__name__}: {e}"
 
    ruta = "ERROR" if error else ruta_de(estado, pendiente)
    fila = {
        "id": mensaje["id"],
        "intencion_esperada": caso["intencion"],
        "intencion": estado.get("intencion"),
        "ruta_esperada": caso["ruta"],
        "ruta": ruta,
        "ruta_ok": ruta == caso["ruta"],
        "error": error,
    }
    if caso["intencion"] == "consulta":
        respuesta = estado.get("respuesta", "")
        fuentes = {f["documento"] for f in estado.get("fuentes", [])}
        fila.update({
            "dato_clave_ok": contiene(respuesta, caso["debe_contener"]),
            "debe_contener": caso["debe_contener"],
            "documento_ok": caso["documento"] in fuentes,
            "respuesta": respuesta,
        })
    if "items" in caso:
        esperados = {(i["sku"], i["cantidad"]) for i in caso["items"]}
        extraidos = {(i["sku"], i["cantidad"]) for i in estado.get("items", [])}
        fila["items_ok"] = extraidos == esperados
    return fila
 
 
def resumir(filas: list[dict]) -> dict:
    consultas = [f for f in filas if "dato_clave_ok" in f]
    pedidos = [f for f in filas if "items_ok" in f]
    return {
        "ramas_correctas": sum(f["ruta_ok"] for f in filas),
        "total": len(filas),
        "dato_clave": sum(f["dato_clave_ok"] for f in consultas),
        "consultas": len(consultas),
        "documento_correcto": sum(f["documento_ok"] for f in consultas),
        "items_exactos": sum(f["items_ok"] for f in pedidos),
        "pedidos": len(pedidos),
    }
 
 
def evaluar(etiqueta: str = "", salida=print) -> dict:
    with open(config.DIR_DATOS / "golden_set.json", encoding="utf-8") as f:
        golden = {c["id"]: c for c in json.load(f)["casos"]}
    with open(config.DIR_DATOS / "bandeja.jsonl", encoding="utf-8") as f:
        mensajes = [json.loads(linea) for linea in f if linea.strip()]
 
    filas = []
    for mensaje in mensajes:
        fila = evaluar_caso(mensaje, golden[mensaje["id"]])
        filas.append(fila)
        marca = "OK " if fila["ruta_ok"] else "MAL"
        extra = ""
        if "dato_clave_ok" in fila:
            extra = f" | dato clave {'OK' if fila['dato_clave_ok'] else 'FALTA'}"
        salida(f"{fila['id']}  {marca}  esperada {fila['ruta_esperada']:<19} obtenida {fila['ruta']}{extra}")
        if fila["error"]:
            salida(f"          error: {fila['error']}")
 
    r = resumir(filas)
    salida("")
    salida(f"RAMA CORRECTA : {r['ramas_correctas']}/{r['total']}")
    salida(f"DATO CLAVE    : {r['dato_clave']}/{r['consultas']} consultas")
    salida(f"(informativo) documento citado correcto: {r['documento_correcto']}/{r['consultas']}"
           f" | items extraidos exactos: {r['items_exactos']}/{r['pedidos']}")
 
    nombre = f"evaluacion_{etiqueta}.json" if etiqueta else "evaluacion.json"
    config.DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_ARTEFACTOS / nombre
    destino.write_text(json.dumps({"metricas": r, "casos": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    salida(f"Detalle guardado en {destino}")
    return r
 