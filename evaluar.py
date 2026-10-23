"""Corre los 20 mensajes de la bandeja contra el golden set y saca los dos numeros.
 
Ejemplos (desde la raiz del proyecto):
    python evaluar.py --retriever denso      # guarda artefactos/evaluacion_denso.json
    python evaluar.py --retriever hibrido    # guarda artefactos/evaluacion_hibrido.json
    python evaluar.py                        # usa RETRIEVER del .env (por defecto hibrido)
"""
import argparse
 
from mesa_pedidos import config
from mesa_pedidos.evaluacion import evaluar
from mesa_pedidos.recuperacion import MODOS
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", choices=MODOS, help="modo de busqueda para esta corrida")
    parser.add_argument("--etiqueta", default="", help="nombre del archivo de salida (por defecto, el retriever)")
    parser.add_argument("--descomponer", choices=("si", "no"), help="partir consultas compuestas en sub-preguntas")
    args = parser.parse_args()
    if args.descomponer:
        config.DESCOMPONER = args.descomponer == "si"
    if args.retriever:
        config.RETRIEVER = args.retriever
    print(f"Retriever: {config.RETRIEVER} | descomponer: {'si' if config.DESCOMPONER else 'no'}\n")
    evaluar(args.etiqueta or config.RETRIEVER)
 