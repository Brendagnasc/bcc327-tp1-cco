"""
geometria.py

Modelagem geométrica implementada pelo grupo: converte os segmentos da
árvore CCO em triângulos para desenhar vasos com espessura proporcional
ao raio.

Modelo 2D: cada segmento vira uma "faixa" (retângulo = 2 triângulos) com
largura 2 * raio, mais um disco em cada ponta (leque de triângulos), que
arredonda as junções e evita frestas nas bifurcações.
"""

import numpy as np


def faixas_2d(pontos, segmentos, meia_largura, n_lados=16):
    """Gera os triângulos das faixas 2D.

    pontos       (N, 3)  coordenadas (usa só x, y)
    segmentos    (M, 2)  índices dos pontos de cada segmento
    meia_largura (M,)    metade da largura de cada faixa (o raio já em escala)
    n_lados              lados do polígono que aproxima cada disco

    Retorna:
    vertices (V, 3) float32, 3 vértices por triângulo, z = 0
    seg_id   (V,)   int, de qual segmento veio cada vértice (para colorir)
    """
    a = pontos[segmentos[:, 0], :2]                   # (M, 2) início
    b = pontos[segmentos[:, 1], :2]                   # (M, 2) fim
    w = meia_largura[:, None]                         # (M, 1)

    # Direção unitária do segmento e normal (direção girada 90 graus)
    d = b - a
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)    # (M, 2)
    desloc = normal * w

    # Retângulo: 4 cantos, dividido em 2 triângulos
    p1, p2 = a + desloc, a - desloc
    p3, p4 = b - desloc, b + desloc
    retangulo = np.stack([np.stack([p1, p2, p3], axis=1),
                          np.stack([p1, p3, p4], axis=1)], axis=1)   # (M, 2, 3, 2)

    # Disco: leque de n_lados triângulos (centro, borda k, borda k+1)
    ang = np.linspace(0.0, 2.0 * np.pi, n_lados + 1)
    circ = np.stack([np.cos(ang), np.sin(ang)], axis=1)               # (n+1, 2)

    def disco(centro):
        borda = centro[:, None, :] + circ[None, :, :] * w[:, :, None]  # (M, n+1, 2)
        c = np.repeat(centro[:, None, :], n_lados, axis=1)             # (M, n, 2)
        return np.stack([c, borda[:, :-1], borda[:, 1:]], axis=2)      # (M, n, 3, 2)

    # Junta tudo por segmento: (M, 2 + 2n, 3, 2)
    tri = np.concatenate([retangulo, disco(a), disco(b)], axis=1)

    # Ordem de desenho: vasos finos primeiro, grossos por cima
    ordem = np.argsort(meia_largura)
    tri = tri[ordem]

    tris_por_seg = tri.shape[1]
    vertices_2d = tri.reshape(-1, 2)
    vertices = np.zeros((len(vertices_2d), 3), dtype=np.float32)
    vertices[:, :2] = vertices_2d
    seg_id = np.repeat(ordem, tris_por_seg * 3)
    return vertices, seg_id
