"""Tools que leen los CSV de la empresa y devuelven datos duros.
 
Se usan desde nodos del grafo (no las llama un agente por su cuenta).
Nunca lanzan excepciones: si algo no existe, devuelven {"encontrado": False, ...}.
"""
import csv
from pathlib import Path
 
from langchain_core.tools import tool
from pydantic import BaseModel, Field
 
DATOS = Path(__file__).resolve().parent.parent / "datos"
 
 
def _leer_csv(nombre: str) -> list[dict]:
    # utf-8-sig: los CSV traen BOM al inicio; sin esto la primera columna
    # se llamaria "\ufeffsku" en vez de "sku".
    with open(DATOS / nombre, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
 
 
# ---------------------------------------------------------------- producto
def _consultar_producto(sku: str) -> dict:
    sku = sku.strip().upper()
    for fila in _leer_csv("catalogo.csv"):
        if fila["sku"].strip().upper() == sku:
            return {
                "encontrado": True,
                "sku": fila["sku"],
                "nombre": fila["nombre"],
                "rubro": fila["rubro"],
                "marca": fila["marca"],
                "unidad": fila["unidad"],
                "precio_lista": float(fila["precio_lista"]),
                "alicuota_iva": float(fila["alicuota_iva"]),
                "stock": int(fila["stock"]),
                "deposito": fila["deposito"],
            }
    return {"encontrado": False, "sku": sku, "error": "SKU inexistente en el catalogo"}
 
 
class ConsultarProductoArgs(BaseModel):
    sku: str = Field(description="Codigo del producto, por ejemplo 'AB-1020'.")
 
 
@tool("consultar_producto", args_schema=ConsultarProductoArgs)
def consultar_producto(sku: str) -> dict:
    """Busca un producto por SKU en el catalogo y devuelve nombre, marca,
    precio de lista (sin IVA), alicuota de IVA, stock y deposito.
    Si el SKU no existe devuelve {"encontrado": False}."""
    return _consultar_producto(sku)
 
 
# ----------------------------------------------------------------- credito
def _verificar_credito(cuit: str, monto_neto: float) -> dict:
    cuit = cuit.strip()
    for fila in _leer_csv("clientes.csv"):
        if fila["cuit"].strip() == cuit:
            limite = float(fila["limite_credito"])
            saldo = float(fila["saldo_actual"])
            disponible = limite - saldo
            return {
                "encontrado": True,
                "cuit": fila["cuit"],
                "razon_social": fila["razon_social"],
                "provincia": fila["provincia"],
                "localidad": fila["localidad"],
                "condicion_iva": fila["condicion_iva"],
                "categoria": fila["categoria"],
                "estado": fila["estado"],
                "mora_dias": int(fila["mora_dias"]),
                "descuento_habitual": float(fila["descuento_habitual"]),
                "limite_credito": limite,
                "saldo_actual": saldo,
                "credito_disponible": disponible,
                "monto_neto_pedido": monto_neto,
                "supera_limite": saldo + monto_neto > limite,
            }
    return {"encontrado": False, "cuit": cuit, "error": "CUIT inexistente en clientes"}
 
 
class VerificarCreditoArgs(BaseModel):
    cuit: str = Field(description="CUIT del cliente con guiones, por ejemplo '30-71455982-2'.")
    monto_neto: float = Field(
        ge=0, description="Monto neto del pedido en pesos, sin IVA."
    )
 
 
@tool("verificar_credito", args_schema=VerificarCreditoArgs)
def verificar_credito(cuit: str, monto_neto: float) -> dict:
    """Busca al cliente por CUIT y devuelve su limite de credito, saldo,
    credito disponible, estado, dias de mora y descuento habitual.
    Indica si saldo + monto_neto supera el limite (art. 7).
    Si el CUIT no existe devuelve {"encontrado": False}."""
    return _verificar_credito(cuit, monto_neto)
 