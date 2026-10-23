"""Pruebas de reglas.py. Cada caso sale de un articulo de Condiciones Comerciales 2026.
 
Datos armados a mano (sin leer los CSV), salvo la ultima prueba que usa la tool real.
Uso: python -m pytest
"""
from mesa_pedidos.reglas import descuento_volumen, evaluar_pedido, tope_descuento
 
 
def cliente(**cambios):
    base = {
        "encontrado": True, "razon_social": "Cliente de prueba", "categoria": "A",
        "condicion_iva": "Responsable Inscripto", "estado": "activo", "mora_dias": 0,
        "descuento_habitual": 10.0, "limite_credito": 5_000_000.0,
        "saldo_actual": 1_000_000.0, "credito_disponible": 4_000_000.0,
    }
    return {**base, **cambios}
 
 
def producto(sku="AB-1020", precio=100_000.0, stock=1000, rubro="Abrasivos", marca="Tyrolit"):
    return {sku: {"encontrado": True, "sku": sku, "nombre": sku, "rubro": rubro,
                  "marca": marca, "precio_lista": precio, "stock": stock, "deposito": "Cordoba"}}
 
 
def pedido(cant=10, **kw_evaluar):
    prods = producto()
    return evaluar_pedido(kw_evaluar.pop("cli", cliente()),
                          [{"sku": "AB-1020", "cantidad": cant}], prods, **kw_evaluar)
 
 
# --- Art. 4: escala por volumen
def test_escala_de_volumen():
    assert descuento_volumen(249_999) == 0
    assert descuento_volumen(999_999) == 0
    assert descuento_volumen(1_000_000) == 5
    assert descuento_volumen(2_999_999) == 5
    assert descuento_volumen(3_000_000) == 8
    assert descuento_volumen(6_000_000) == 12
 
 
# --- Art. 5: topes
def test_tope_15_solo_en_herramienta_electrica_de_esas_marcas():
    assert tope_descuento({"rubro": "Herramienta electrica", "marca": "Dewalt"}) == 15
    assert tope_descuento({"rubro": "Herramienta electrica", "marca": "Makita"}) == 15
    assert tope_descuento({"rubro": "Herramienta electrica", "marca": "Lusqtoff"}) == 20
    assert tope_descuento({"rubro": "Fijaciones", "marca": "Bosch"}) == 20
 
 
def test_descuento_total_se_corta_en_el_tope():
    # habitual 15% + volumen 8% = 23% -> se aplica 20% (art. 4 y 5)
    r = pedido(cant=40, cli=cliente(descuento_habitual=15.0))  # lista $4.000.000
    assert r["renglones"][0]["descuento_pct"] == 20
 
 
def test_descuento_solicitado_mayor_a_20_va_a_una_persona():
    assert pedido(descuento_solicitado=25)["decision"] == "humano"
    assert pedido(descuento_solicitado=20)["decision"] == "confirmar"
 
 
# --- Art. 6: pago anticipado
def test_pago_anticipado_descuenta_3_por_ciento():
    normal = pedido()["neto"]
    anticipado = pedido(pago_anticipado=True)["neto"]
    assert anticipado == round(normal * 0.97, 2)
 
 
# --- Art. 7: credito y mora
def test_dentro_del_limite_se_confirma():
    assert pedido()["decision"] == "confirmar"
 
 
def test_supera_el_limite_va_a_una_persona():
    assert pedido(cli=cliente(credito_disponible=100_000.0))["decision"] == "humano"
 
 
def test_pago_anticipado_no_consume_credito():
    r = pedido(cli=cliente(credito_disponible=100_000.0), pago_anticipado=True)
    assert r["decision"] == "confirmar"
 
 
def test_mora_entre_30_y_60_exige_autorizacion():
    assert pedido(cli=cliente(mora_dias=38))["decision"] == "humano"
    assert pedido(cli=cliente(mora_dias=60))["decision"] == "humano"
 
 
def test_mora_mayor_a_60_se_rechaza_salvo_pago_anticipado():
    suspendido = cliente(mora_dias=95, estado="suspendido")
    assert pedido(cli=suspendido)["decision"] == "rechazar"
    assert pedido(cli=suspendido, pago_anticipado=True)["decision"] == "confirmar"
 
 
# --- Art. 9: stock
def test_sin_stock_va_a_una_persona():
    r = evaluar_pedido(cliente(), [{"sku": "AB-1020", "cantidad": 60}], producto(stock=52))
    assert r["decision"] == "humano"
    assert r["renglones"][0]["stock_ok"] is False
 
 
# --- Datos que faltan
def test_sku_inexistente_se_deriva():
    r = evaluar_pedido(cliente(), [{"sku": "XX-0000", "cantidad": 1}],
                       {"XX-0000": {"encontrado": False}})
    assert r["decision"] == "derivar"
 
 
def test_sin_items_o_sin_cliente_se_deriva():
    assert evaluar_pedido(cliente(), [], {})["decision"] == "derivar"
    assert evaluar_pedido({"encontrado": False}, [{"sku": "A", "cantidad": 1}], {})["decision"] == "derivar"
 
 
# --- Art. 3: pedido minimo
def test_pedido_por_debajo_del_minimo_va_a_una_persona():
    assert pedido(cant=1)["decision"] == "humano"  # $90.000 netos < $250.000
 
 
# --- Integracion con la tool real
def test_la_tool_devuelve_rubro_y_marca():
    from mesa_pedidos.herramientas import consultar_producto
    p = consultar_producto.invoke({"sku": "HE-4110"})
    assert tope_descuento(p) == 15