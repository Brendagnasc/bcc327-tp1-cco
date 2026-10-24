"""
rasterizacao.py

Algoritmos clássicos de rasterização implementados pelo grupo.
Eles não desenham nada: só calculam QUAIS pixels (coordenadas inteiras)
devem ser acesos. Quem pinta é o modo didático.

- bresenham: só aritmética inteira, variável de decisão p
- dda: incremento em ponto flutuante + arredondamento (para comparação)
- scanline_fill: preenchimento de polígonos (convexos ou côncavos) com
  tabela de arestas (ET) e tabela de arestas ativas (AET)
"""

import math


def bresenham(x0, y0, x1, y1, rastro=None):
    """Retorna a lista de pixels do segmento (x0,y0)-(x1,y1), em qualquer octante.

    Ideia (caso |inclinação| <= 1): x avança sempre 1; a cada passo decide-se
    se y fica ou sobe. A variável de decisão p é proporcional à diferença
    entre as distâncias da reta ideal aos dois pixels candidatos:
        p0 = 2*dy - dx
        p < 0   -> mantém y,  p += 2*dy
        p >= 0  -> y += 1,    p += 2*dy - 2*dx
    Para |inclinação| > 1 os papéis de x e y se invertem. Os sinais sx e sy
    cobrem os outros octantes.

    Se 'rastro' for uma lista, recebe (k, p_k, x, y) de cada passo.
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    x, y = x0, y0
    pixels = []

    if dx >= dy:                       # eixo dominante: x
        p = 2 * dy - dx
        for k in range(dx + 1):
            pixels.append((x, y))
            if rastro is not None:
                rastro.append((k, p, x, y))
            if p >= 0:
                y += sy
                p += 2 * dy - 2 * dx
            else:
                p += 2 * dy
            x += sx
    else:                              # eixo dominante: y
        p = 2 * dx - dy
        for k in range(dy + 1):
            pixels.append((x, y))
            if rastro is not None:
                rastro.append((k, p, x, y))
            if p >= 0:
                x += sx
                p += 2 * dx - 2 * dy
            else:
                p += 2 * dx
            y += sy

    return pixels


def dda(x0, y0, x1, y1):
    """Digital Differential Analyzer: anda 'passos' vezes somando incrementos
    reais em x e y e arredonda cada posição para o pixel mais próximo."""
    passos = max(abs(x1 - x0), abs(y1 - y0))
    if passos == 0:
        return [(x0, y0)]
    inc_x = (x1 - x0) / passos
    inc_y = (y1 - y0) / passos
    x, y = float(x0), float(y0)
    pixels = []
    for _ in range(passos + 1):
        pixels.append((int(x + 0.5) if x >= 0 else -int(-x + 0.5),
                       int(y + 0.5) if y >= 0 else -int(-y + 0.5)))
        x += inc_x
        y += inc_y
    return pixels


def scanline_linhas(vertices):
    """Scanline Fill: calcula, linha a linha, os trechos (spans) internos do polígono.

    vertices: lista [(x, y), ...] em ordem; o último liga no primeiro.

    1. Tabela de arestas (ET): cada aresta não horizontal é guardada no
       "balde" da linha onde começa (y mínimo), com [y_max, x_atual, 1/m].
    2. Para cada linha y, de baixo para cima:
       - entram na AET as arestas que começam em y;
       - saem as que já terminaram (y_max <= y);
       - os x das arestas ativas são ordenados e preenchidos aos pares
         (regra par-ímpar: entra no polígono, sai, entra, sai...);
       - cada x avança 1/m para a próxima linha (coerência entre linhas).
    O intervalo [y_min, y_max) é semiaberto: um vértice comum a duas arestas
    conta uma vez só, e as arestas horizontais são ignoradas.

    Retorna lista de (y, interseções x, spans [(x_ini, x_fim), ...]).
    """
    tabela = {}
    n = len(vertices)
    for i in range(n):
        (xa, ya), (xb, yb) = vertices[i], vertices[(i + 1) % n]
        if ya == yb:
            continue                                   # horizontal: ignora
        if ya > yb:
            xa, ya, xb, yb = xb, yb, xa, ya            # garante ya < yb
        inv_m = (xb - xa) / (yb - ya)
        y_ini = math.ceil(ya)
        x_ini = xa + (y_ini - ya) * inv_m              # x na primeira linha
        tabela.setdefault(y_ini, []).append([yb, x_ini, inv_m])

    if not tabela:
        return []

    linhas = []
    ativas = []
    y_max = max(a[0] for balde in tabela.values() for a in balde)
    for y in range(min(tabela), math.ceil(y_max)):
        ativas.extend(tabela.get(y, []))
        ativas = [a for a in ativas if a[0] > y]
        ativas.sort(key=lambda a: a[1])
        xs = [a[1] for a in ativas]
        spans = [(math.ceil(xs[k]), math.floor(xs[k + 1]))
                 for k in range(0, len(xs) - 1, 2)]
        linhas.append((y, xs, spans))
        for a in ativas:
            a[1] += a[2]
    return linhas


def scanline_fill(vertices):
    """Lista de pixels internos do polígono (usa scanline_linhas)."""
    return [(x, y) for y, _, spans in scanline_linhas(vertices)
            for xi, xf in spans for x in range(xi, xf + 1)]


def imprimir_scanline(vertices):
    """Tabela linha a linha do Scanline Fill (exemplo para o relatório)."""
    print(f"Scanline Fill do polígono {vertices}")
    print(f"{'y':>4}   {'interseções x':<28} spans preenchidos")
    for y, xs, spans in scanline_linhas(vertices):
        txt_x = ", ".join(f"{x:.2f}" for x in xs)
        txt_s = "  ".join(f"[{a}..{b}]" for a, b in spans if a <= b)
        print(f"{y:>4}   {txt_x:<28} {txt_s}")


def imprimir_rastro(x0, y0, x1, y1):
    """Tabela passo a passo do Bresenham (exemplo numérico para o relatório)."""
    rastro = []
    bresenham(x0, y0, x1, y1, rastro)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    print(f"Bresenham de ({x0},{y0}) a ({x1},{y1}):  dx = {dx}, dy = {dy}, "
          f"2dy = {2 * dy}, 2dy - 2dx = {2 * dy - 2 * dx}")
    print(f"{'k':>3} {'p_k':>5}   pixel")
    for k, p, x, y in rastro:
        print(f"{k:>3} {p:>5}   ({x}, {y})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        imprimir_rastro(*map(int, sys.argv[1:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "scanline":
        imprimir_scanline([(2, 1), (12, 3), (9, 7), (12, 11), (5, 9), (2, 11)])
    else:
        imprimir_rastro(20, 10, 30, 18)    # exemplo clássico (Hearn & Baker)
