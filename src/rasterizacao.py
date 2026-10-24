"""
rasterizacao.py

Algoritmos clássicos de rasterização implementados pelo grupo.
Eles não desenham nada: só calculam QUAIS pixels (coordenadas inteiras)
devem ser acesos. Quem pinta é o modo didático.

- bresenham: só aritmética inteira, variável de decisão p
- dda: incremento em ponto flutuante + arredondamento (para comparação)
"""


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
    else:
        imprimir_rastro(20, 10, 30, 18)    # exemplo clássico (Hearn & Baker)
