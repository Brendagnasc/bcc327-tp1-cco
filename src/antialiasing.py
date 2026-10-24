"""
antialiasing.py

Supersampling (SSAA) implementado pelo grupo, usando os próprios
algoritmos de rasterização (Bresenham e Scanline Fill).

Ideia: renderizar a cena num framebuffer k vezes maior em cada direção
e depois reduzir, fazendo a média de cada bloco k x k de subpixels.
Um pixel da borda que é metade coberto pela linha fica com 50% da cor:
o "degrau" vira uma transição suave.

A cena é descrita em coordenadas contínuas da grade final:
    ("linha", (x0, y0, x1, y1), cor)    segmento com 1 pixel de largura
    ("poligono", [(x, y), ...], cor)    polígono preenchido
"""

import numpy as np

from rasterizacao import bresenham, scanline_fill


def _para_alta(v, k):
    """Coordenada da grade final -> grade k vezes maior (centros alinhados)."""
    return (v + 0.5) * k - 0.5


def renderizar(cena, colunas, linhas, k=1, fundo=255):
    """Renderiza a cena com fator de supersampling k (k = 1: sem antialiasing)."""
    alta = np.full((linhas * k, colunas * k, 3), fundo, dtype=float)
    a_lin, a_col = alta.shape[:2]

    # deslocamentos do "carimbo" k x k: uma linha de 1 pixel final = k subpixels
    offs = np.arange(k) - k // 2

    for tipo, geo, cor in cena:
        if tipo == "poligono":
            verts = [(_para_alta(x, k), _para_alta(y, k)) for x, y in geo]
            px = np.array(scanline_fill(verts), dtype=int).reshape(-1, 2)
        else:
            x0, y0, x1, y1 = (int(round(_para_alta(v, k))) for v in geo)
            base = np.array(bresenham(x0, y0, x1, y1), dtype=int)
            # carimba um quadrado k x k em cada pixel da linha fina
            px = (base[:, None, None, :]
                  + np.stack(np.meshgrid(offs, offs, indexing="ij"), axis=-1)[None])
            px = px.reshape(-1, 2)

        dentro = (px[:, 0] >= 0) & (px[:, 0] < a_col) & (px[:, 1] >= 0) & (px[:, 1] < a_lin)
        px = px[dentro]
        alta[px[:, 1], px[:, 0]] = cor

    # redução: média de cada bloco k x k (filtro caixa)
    baixa = alta.reshape(linhas, k, colunas, k, 3).mean(axis=(1, 3))
    return np.rint(baixa).astype(np.uint8)


def cena_estrela(colunas, linhas):
    """Cena de teste com alta frequência espacial: leque de linhas inclinadas
    saindo de um centro, mais um triângulo de bordas oblíquas."""
    cx, cy, r = colunas / 2 - 0.5, linhas * 0.68, min(colunas, linhas) * 0.42
    cena = []
    for ang in np.radians(np.arange(0, 360, 12)):
        cena.append(("linha", (cx, cy, cx + r * np.cos(ang), cy + r * np.sin(ang)),
                     (20, 20, 20)))
    tri = [(colunas * 0.12, linhas * 0.06), (colunas * 0.9, linhas * 0.14),
           (colunas * 0.35, linhas * 0.33)]
    cena.append(("poligono", tri, (200, 90, 0)))
    return cena
