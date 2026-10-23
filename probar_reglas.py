"""Prueba manual de reglas.py con los 9 pedidos de la bandeja (sin LLM, sin API).
 
Los items salen del texto de cada mensaje; pago_anticipado y descuento_solicitado
tambien (los que el LLM extraera mas adelante). Uso: python probar_reglas.py
"""
from mesa_pedidos.herramientas import consultar_producto, verificar_credito
from mesa_pedidos.reglas import evaluar_pedido
 
PEDIDOS = {
    "MSG-009": ("30-71455982-2", {"AB-1020": 200, "AB-1120": 50, "ME-6030": 10}, {}),
    "MSG-010": ("30-69874512-2", {"HE-4110": 4, "HE-4500": 2, "SE-5010": 30, "SE-5055": 100}, {}),
    "MSG-011": ("20-31447856-9", {"SE-5025": 20, "ME-6030": 15}, {"pago_anticipado": True}),
    "MSG-012": ("30-70551239-2", {"LU-8010": 6, "LU-8090": 10, "FI-7075": 2}, {}),
    "MSG-013": ("30-70988341-7", {"HE-4180": 3, "NE-9010": 1, "NE-9055": 4}, {}),
    "MSG-014": ("30-71622074-1", {"SU-3041": 2, "SU-3150": 20}, {}),
    "MSG-015": ("30-71099873-2", {"ME-6080": 5, "HE-4410": 8}, {}),
    "MSG-016": ("30-71340098-6", {"AB-1035": 500, "AB-1120": 300, "HE-4110": 60}, {"descuento_solicitado": 25}),
    "MSG-017": ("30-71788450-3", {"SE-5090": 10, "SU-3200": 20}, {}),
}
 
for msg_id, (cuit, lineas, extra) in PEDIDOS.items():
    cliente = verificar_credito.invoke({"cuit": cuit, "monto_neto": 0})
    items = [{"sku": sku, "cantidad": cant} for sku, cant in lineas.items()]
    productos = {sku: consultar_producto.invoke({"sku": sku}) for sku in lineas}
    r = evaluar_pedido(cliente, items, productos, **extra)
    print(f"{msg_id}  {r['decision'].upper():9}  neto ${r['neto']:>14,.0f}  ({cliente['razon_social']})")
    for alerta in r["alertas"]:
        print(f"           - {alerta}")
 