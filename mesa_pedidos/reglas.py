"""Reglas comerciales de Condiciones Comerciales 2026.
 
Funciones puras: no usan LLM ni leen archivos. Reciben datos ya consultados
(por las tools) y devuelven una decision. Cada constante cita su articulo.
"""
 
# --- Pedido minimo (art. 3), en pesos sin IVA
MINIMO_CATEGORIA_AB = 250_000
MINIMO_CATEGORIA_C = 120_000  # categoria C y monotributistas
 
# --- Escala de descuento por volumen (art. 4): (desde, % de descuento)
ESCALA_VOLUMEN = [(6_000_000, 12.0), (3_000_000, 8.0), (1_000_000, 5.0), (0, 0.0)]
 
# --- Tope de descuento (art. 5)
TOPE_GENERAL = 20.0
TOPE_ELECTRICAS_MARCAS = 15.0  # herramienta electrica Dewalt, Makita y Bosch
MARCAS_TOPE_REDUCIDO = {"dewalt", "makita", "bosch"}
RUBRO_ELECTRICO = "herramienta electrica"
 
# --- Pago anticipado (art. 6): descuento financiero, fuera del tope del art. 5
DESCUENTO_PAGO_ANTICIPADO = 3.0
 
# --- Mora (art. 7)
MORA_SUSPENSION = 60       # mas de 60 dias: suspendido
MORA_AUTORIZACION = 30     # de 30 a 60 dias: cada pedido requiere autorizacion
 
 
def _pesos(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")
 
 
def descuento_volumen(subtotal_lista: float) -> float:
    """Art. 4: % de descuento segun el monto del pedido a precio de lista."""
    for desde, porcentaje in ESCALA_VOLUMEN:
        if subtotal_lista >= desde:
            return porcentaje
    return 0.0
 
 
def tope_descuento(producto: dict) -> float:
    """Art. 5: 15% para herramienta electrica Dewalt/Makita/Bosch; 20% el resto."""
    es_electrica = producto.get("rubro", "").strip().lower() == RUBRO_ELECTRICO
    if es_electrica and producto.get("marca", "").strip().lower() in MARCAS_TOPE_REDUCIDO:
        return TOPE_ELECTRICAS_MARCAS
    return TOPE_GENERAL
 
 
def _resultado(decision, renglones=None, subtotal=0.0, pct_vol=0.0, neto=0.0, alertas=None):
    return {
        "decision": decision,
        "renglones": renglones or [],
        "subtotal_lista": round(subtotal, 2),
        "descuento_volumen_pct": pct_vol,
        "neto": round(neto, 2),
        "alertas": alertas or [],
    }
 
 
def evaluar_pedido(
    cliente: dict,
    items: list[dict],
    productos: dict[str, dict],
    descuento_solicitado: float | None = None,
    pago_anticipado: bool = False,
) -> dict:
    """Decide si el pedido se confirma, pasa a una persona, se rechaza o se deriva.
 
    cliente: resultado de verificar_credito. productos: {sku: resultado de consultar_producto}.
    decision: "derivar" | "rechazar" | "humano" | "confirmar".
    """
    # --- Datos que faltan: no es un pedido que se pueda evaluar
    if not cliente or not cliente.get("encontrado"):
        return _resultado("derivar", alertas=["Cliente no encontrado en el sistema."])
    if not items:
        return _resultado("derivar", alertas=["No se pudo extraer ningun producto del mensaje."])
    inexistentes = [
        i["sku"] for i in items if not productos.get(i["sku"].strip().upper(), {}).get("encontrado")
    ]
    if inexistentes:
        return _resultado("derivar", alertas=[f"SKU inexistente en el catalogo: {', '.join(inexistentes)}."])
 
    # --- Precios y descuentos (arts. 4 y 5)
    precios = [(i, productos[i["sku"].strip().upper()]) for i in items]
    subtotal_lista = sum(p["precio_lista"] * i["cantidad"] for i, p in precios)
    pct_vol = descuento_volumen(subtotal_lista)
    habitual = cliente["descuento_habitual"]
 
    alertas: list[str] = []
    renglones = []
    for item, prod in precios:
        pct = min(habitual + pct_vol, tope_descuento(prod))  # acumulable hasta el tope
        subtotal = prod["precio_lista"] * item["cantidad"] * (1 - pct / 100)
        stock_ok = item["cantidad"] <= prod["stock"]
        if not stock_ok:  # art. 9: nunca se confirma con stock que no existe
            alertas.append(
                f"Sin stock de {prod['sku']}: piden {item['cantidad']}, hay {prod['stock']} (art. 9)."
            )
        renglones.append({
            "sku": prod["sku"], "nombre": prod["nombre"], "cantidad": item["cantidad"],
            "precio_lista": prod["precio_lista"], "descuento_pct": pct,
            "subtotal": round(subtotal, 2), "stock": prod["stock"],
            "deposito": prod["deposito"], "stock_ok": stock_ok,
        })
 
    neto = sum(r["subtotal"] for r in renglones)
    if pago_anticipado:  # art. 6
        neto *= 1 - DESCUENTO_PAGO_ANTICIPADO / 100
 
    # --- Descuento pedido por el cliente (art. 5)
    if descuento_solicitado is not None and descuento_solicitado > TOPE_GENERAL:
        alertas.append(
            f"Pide {descuento_solicitado:g}% de descuento; el tope es {TOPE_GENERAL:g}% "
            "y requiere autorizacion de la Gerencia Comercial (art. 5)."
        )
 
    # --- Pedido minimo (art. 3)
    es_c = cliente["categoria"] == "C" or cliente["condicion_iva"] == "Monotributo"
    minimo = MINIMO_CATEGORIA_C if es_c else MINIMO_CATEGORIA_AB
    if neto < minimo:
        alertas.append(
            f"Neto {_pesos(neto)} por debajo del minimo {_pesos(minimo)}: "
            "queda como pedido en espera (art. 3)."
        )
 
    # --- Cuenta corriente para categoria C (art. 6)
    if cliente["categoria"] == "C" and not pago_anticipado:
        alertas.append("Categoria C sin pago anticipado: la cuenta corriente requiere autorizacion (art. 6).")
 
    # --- Mora (art. 7)
    mora = cliente["mora_dias"]
    if (cliente["estado"] == "suspendido" or mora > MORA_SUSPENSION) and not pago_anticipado:
        alertas.insert(0, (
            f"Cliente suspendido con {mora} dias de mora: no se toma pedido en cuenta "
            "corriente, solo con pago anticipado (art. 7)."
        ))
        return _resultado("rechazar", renglones, subtotal_lista, pct_vol, neto, alertas)
    if MORA_AUTORIZACION <= mora <= MORA_SUSPENSION:
        alertas.append(
            f"Cliente con {mora} dias de mora: cada pedido requiere autorizacion (art. 7)."
        )
 
    # --- Limite de credito (art. 7): el pago anticipado no consume credito
    if not pago_anticipado and neto > cliente["credito_disponible"]:
        alertas.append(
            f"Supera el limite de credito: saldo {_pesos(cliente['saldo_actual'])} + pedido "
            f"{_pesos(neto)} > limite {_pesos(cliente['limite_credito'])} (art. 7)."
        )
 
    decision = "humano" if alertas else "confirmar"
    return _resultado(decision, renglones, subtotal_lista, pct_vol, neto, alertas)
 