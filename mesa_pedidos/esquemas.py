"""Formas que el LLM esta obligado a devolver (structured output con Pydantic)."""
from typing import Literal
 
from pydantic import BaseModel, Field
 
 
class Clasificacion(BaseModel):
    intencion: Literal["consulta", "pedido", "derivar"] = Field(
        description=(
            "consulta: pregunta sobre politicas, fletes, garantias, productos. "
            "pedido: el cliente pide mercaderia concreta. "
            "derivar: reclamo, oferta de proveedor, o pedido de asesoramiento "
            "que una persona debe atender."
        )
    )
    motivo: str = Field(description="Una frase que justifica la clasificacion.")
 
 
class ItemPedido(BaseModel):
    sku: str = Field(description="Codigo del producto tal como figura en el mensaje, ej. 'AB-1020'.")
    cantidad: int = Field(gt=0, description="Unidades pedidas.")
 
 
class PedidoExtraido(BaseModel):
    items: list[ItemPedido] = Field(
        description="Un renglon por cada producto con SKU explicito en el mensaje."
    )
    descuento_solicitado: float | None = Field(
        default=None,
        description="Porcentaje de descuento que el cliente pide, ej. 25 para '25%'. None si no pide.",
    )
    pago_anticipado: bool = Field(
        default=False,
        description="True solo si el cliente dice que paga antes del envio (ej. transferencia anticipada).",
    )
 
 
class SubPreguntas(BaseModel):
    preguntas: list[str] = Field(
        description=(
            "La consulta separada en preguntas independientes, una por tema distinto. "
            "Si la consulta trata un solo tema, una unica pregunta. Maximo 3."
        )
    )
 