"""
leitor_vtk.py

Leitor próprio de arquivos VTK legado (ASCII, DATASET POLYDATA) para os
modelos de árvores arteriais CCO fornecidos na disciplina BCC327.

Implementado pelo grupo, sem uso da biblioteca VTK.

Estrutura esperada do arquivo:
    POINTS     N float      -> N linhas com x y z
    LINES      M total      -> M linhas "2 i j" (um segmento por vaso)
    CELL_DATA  M
    scalars raio float      -> um raio por segmento (atributo de célula, em mm)
    LOOKUP_TABLE default
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ArvoreCCO:
    nome: str
    pontos: np.ndarray      # (N, 3) coordenadas x, y, z
    segmentos: np.ndarray   # (M, 2) índices dos dois pontos de cada segmento
    raios: np.ndarray       # (M,)   raio de cada segmento, em mm

    @property
    def n_pontos(self) -> int:
        return len(self.pontos)

    @property
    def n_segmentos(self) -> int:
        return len(self.segmentos)

    @property
    def eh_3d(self) -> bool:
        return not np.allclose(self.pontos[:, 2], 0.0)


def ler_vtk(caminho) -> ArvoreCCO:
    caminho = Path(caminho)
    linhas = caminho.read_text().splitlines()

    # Cabeçalho: linha 0 = versão, linha 1 = título, linha 2 = formato, linha 3 = tipo
    if not linhas[0].startswith("# vtk DataFile"):
        raise ValueError(f"{caminho.name}: não é um arquivo VTK legado")
    if linhas[2].strip().upper() != "ASCII":
        raise ValueError(f"{caminho.name}: apenas o formato ASCII é suportado")
    if linhas[3].split() != ["DATASET", "POLYDATA"]:
        raise ValueError(f"{caminho.name}: esperado DATASET POLYDATA")

    # Do corpo em diante, tudo vira uma lista de tokens percorrida por um índice
    tokens = " ".join(linhas[4:]).split()
    i = 0
    pontos = segmentos = raios = None

    while i < len(tokens):
        chave = tokens[i].upper()

        if chave == "POINTS":
            n = int(tokens[i + 1])
            i += 3                                   # pula "POINTS n float"
            pontos = np.array(tokens[i:i + 3 * n], dtype=float).reshape(n, 3)
            i += 3 * n

        elif chave == "LINES":
            n_celulas = int(tokens[i + 1])
            i += 3                                   # pula "LINES n total"
            pares = []
            for _ in range(n_celulas):
                k = int(tokens[i])                   # quantos pontos tem a célula
                ids = [int(t) for t in tokens[i + 1:i + 1 + k]]
                # uma polilinha com k pontos vira k-1 segmentos consecutivos
                pares.extend(zip(ids[:-1], ids[1:]))
                i += 1 + k
            segmentos = np.array(pares, dtype=int)

        elif chave == "CELL_DATA":
            m = int(tokens[i + 1])
            i += 2

        elif chave == "SCALARS":
            i += 3                                   # pula "scalars nome tipo"
            if tokens[i].upper() != "LOOKUP_TABLE":  # numComp é opcional
                i += 1
            i += 2                                   # pula "LOOKUP_TABLE default"
            raios = np.array(tokens[i:i + m], dtype=float)
            i += m

        else:
            i += 1

    # Validações básicas de consistência
    if pontos is None or segmentos is None or raios is None:
        raise ValueError(f"{caminho.name}: faltam POINTS, LINES ou o atributo de raio")
    if segmentos.max() >= len(pontos) or segmentos.min() < 0:
        raise ValueError(f"{caminho.name}: segmento referencia ponto inexistente")
    if len(raios) != len(segmentos):
        raise ValueError(f"{caminho.name}: número de raios diferente do de segmentos")

    return ArvoreCCO(caminho.stem, pontos, segmentos, raios)


def resumo(arvore: ArvoreCCO) -> str:
    return (f"{arvore.nome:<20} {'3D' if arvore.eh_3d else '2D':<4} "
            f"{arvore.n_pontos:>7} {arvore.n_segmentos:>10} "
            f"{arvore.raios.min():>11.4f} {arvore.raios.max():>11.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python src/leitor_vtk.py <arquivo.vtk> [outros.vtk ...]")
        sys.exit(1)

    print(f"{'modelo':<20} {'dim':<4} {'pontos':>7} {'segmentos':>10} "
          f"{'raio mín':>11} {'raio máx':>11}")
    for arquivo in sys.argv[1:]:
        print(resumo(ler_vtk(arquivo)))
