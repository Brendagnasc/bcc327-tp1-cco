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


def _esfera_unitaria(n_setores=8, n_aneis=4):
    """Triângulos de uma esfera de raio 1 por latitude/longitude: (T, 3, 3)."""
    theta = np.linspace(0, np.pi, n_aneis + 1)          # polo norte -> polo sul
    phi = np.linspace(0, 2 * np.pi, n_setores + 1)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    grade = np.stack([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)], axis=-1)

    a, b = grade[:-1, :-1], grade[1:, :-1]               # cada "quadradinho" da grade
    c, d = grade[1:, 1:], grade[:-1, 1:]
    tri = np.concatenate([np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
                          np.stack([a, c, d], axis=2).reshape(-1, 3, 3)])
    return tri


def tubos_3d(pontos, segmentos, raio, n_lados=12):
    """Gera cilindros (tubos) 3D para cada segmento, mais uma esfera em cada
    ponto da árvore para fechar as junções das bifurcações.

    pontos    (N, 3)  coordenadas
    segmentos (M, 2)  índices dos pontos de cada segmento
    raio      (M,)    raio de cada tubo, já em escala
    n_lados           lados do polígono da seção transversal

    Retorna:
    vertices (V, 3) float32, 3 vértices por triângulo
    normais  (V, 3) float32, normal de cada vértice (para a iluminação)
    seg_id   (V,)   int, de qual segmento veio cada vértice (para colorir)
    """
    a = pontos[segmentos[:, 0]]
    b = pontos[segmentos[:, 1]]
    r = raio[:, None, None]

    # Base ortonormal (d, e1, e2) de cada segmento: d ao longo do eixo,
    # e1 e e2 formam o plano da seção transversal (o círculo)
    d = b - a
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    auxiliar = np.where(np.abs(d[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    e1 = np.cross(d, auxiliar)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)

    ang = np.linspace(0, 2 * np.pi, n_lados + 1)
    radial = (np.cos(ang)[None, :, None] * e1[:, None, :]
              + np.sin(ang)[None, :, None] * e2[:, None, :])       # (M, n+1, 3)
    anel_a = a[:, None, :] + r * radial
    anel_b = b[:, None, :] + r * radial

    # Cada lado do tubo é um retângulo = 2 triângulos
    k, k1 = slice(0, n_lados), slice(1, n_lados + 1)
    tri_v = np.stack([np.stack([anel_a[:, k], anel_b[:, k], anel_b[:, k1]], axis=2),
                      np.stack([anel_a[:, k], anel_b[:, k1], anel_a[:, k1]], axis=2)], axis=2)
    tri_n = np.stack([np.stack([radial[:, k], radial[:, k], radial[:, k1]], axis=2),
                      np.stack([radial[:, k], radial[:, k1], radial[:, k1]], axis=2)], axis=2)
    tubo_v = tri_v.reshape(-1, 3)
    tubo_n = tri_n.reshape(-1, 3)
    tubo_id = np.repeat(np.arange(len(segmentos)), n_lados * 2 * 3)

    # Esferas nos pontos: cada ponto recebe o raio e a cor do MAIOR segmento
    # que toca nele. Percorrendo os segmentos do mais fino ao mais grosso,
    # a última atribuição (a do mais grosso) é a que fica.
    ordem = np.argsort(raio)
    seg_do_ponto = np.zeros(len(pontos), dtype=int)
    seg_do_ponto[segmentos[ordem, 0]] = ordem
    seg_do_ponto[segmentos[ordem, 1]] = ordem
    usados = np.unique(segmentos)

    unit = _esfera_unitaria()                                       # (T, 3, 3)
    centros = pontos[usados][:, None, None, :]
    raios_esf = raio[seg_do_ponto[usados]][:, None, None, None]
    esf_v = (centros + raios_esf * unit[None]).reshape(-1, 3)
    esf_n = np.broadcast_to(unit[None], (len(usados),) + unit.shape).reshape(-1, 3)
    esf_id = np.repeat(seg_do_ponto[usados], unit.shape[0] * 3)

    vertices = np.concatenate([tubo_v, esf_v]).astype(np.float32)
    normais = np.concatenate([tubo_n, esf_n]).astype(np.float32)
    seg_id = np.concatenate([tubo_id, esf_id])
    return vertices, normais, seg_id
