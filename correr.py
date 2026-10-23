"""Procesa un mensaje de la bandeja y maneja el pedido de aprobacion.
 
Ejemplos (desde la raiz del proyecto):
    python correr.py MSG-002                        # una consulta
    python correr.py MSG-014                        # un pedido que frena: pide decision
    python correr.py MSG-014 --resolver aprobar     # decidir sin preguntar
    python correr.py MSG-014 --reiniciar            # borrar su historia y empezar de cero
"""
import argparse
 
from mesa_pedidos.ejecucion import ejecutar
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mensaje_id", help="id del mensaje de la bandeja, ej. MSG-014")
    parser.add_argument("--resolver", choices=["aprobar", "rechazar"],
                        help="decidir un pedido pendiente sin preguntar")
    parser.add_argument("--motivo", default="", help="motivo de la decision")
    parser.add_argument("--reiniciar", action="store_true",
                        help="borrar el historial de ese mensaje y procesarlo de nuevo")
    args = parser.parse_args()
    ejecutar(args.mensaje_id, resolver=args.resolver, motivo=args.motivo, reiniciar=args.reiniciar)
 