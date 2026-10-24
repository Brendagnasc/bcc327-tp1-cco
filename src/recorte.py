"""
recorte.py

Algoritmos clássicos de recorte (clipping) implementados pelo grupo.

Segmentos de reta contra uma janela retangular alinhada aos eixos
janela = (xmin, ymin, xmax, ymax):
- cohen_sutherland: códigos de região de 4 bits + recorte iterativo
- liang_barsky: forma paramétrica P(t) = P0 + t (P1 - P0), t em [0, 1]

Cada função retorna o segmento recortado (x0, y0, x1, y1) ou None se
o segmento estiver totalmente fora.

Polígonos contra uma janela convexa qualquer (retângulo incluído):
- sutherland_hodgman: recorta o polígono sujeito borda a borda da janela
"""

import random
import time

# Bits do código de região (Cohen–Sutherland)
DENTRO, ESQUERDA, DIREITA, BAIXO, CIMA = 0, 1, 2, 4, 8


def codigo_regiao(x, y, janela):
    """Código de 4 bits: em quais semiplanos externos o ponto está.

        1001 | 1000 | 1010
        -----+------+-----
        0001 | 0000 | 0010
        -----+------+-----
        0101 | 0100 | 0110
    """
    xmin, ymin, xmax, ymax = janela
    c = DENTRO
    if x < xmin:
        c |= ESQUERDA
    elif x > xmax:
        c |= DIREITA
    if y < ymin:
        c |= BAIXO
    elif y > ymax:
        c |= CIMA
    return c


def cohen_sutherland(x0, y0, x1, y1, janela, rastro=None):
    """Recorte de Cohen–Sutherland.

    - c0 | c1 == 0: os dois pontos dentro -> aceite trivial
    - c0 & c1 != 0: os dois do mesmo lado de fora -> rejeição trivial
    - senão: escolhe um ponto de fora, troca-o pela interseção com a
      borda indicada por um bit do seu código e repete.
    """
    xmin, ymin, xmax, ymax = janela
    c0 = codigo_regiao(x0, y0, janela)
    c1 = codigo_regiao(x1, y1, janela)

    while True:
        if rastro is not None:
            rastro.append((x0, y0, c0, x1, y1, c1))
        if not (c0 | c1):
            return (x0, y0, x1, y1)
        if c0 & c1:
            return None

        fora = c0 if c0 else c1
        if fora & CIMA:
            x, y = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0), ymax
        elif fora & BAIXO:
            x, y = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0), ymin
        elif fora & DIREITA:
            x, y = xmax, y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
        else:
            x, y = xmin, y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)

        if fora == c0:
            x0, y0 = x, y
            c0 = codigo_regiao(x0, y0, janela)
        else:
            x1, y1 = x, y
            c1 = codigo_regiao(x1, y1, janela)


def liang_barsky(x0, y0, x1, y1, janela, rastro=None):
    """Recorte de Liang–Barsky.

    Um ponto P(t) está dentro se, para cada borda k, p_k * t <= q_k:
        esquerda: p = -dx, q = x0 - xmin
        direita:  p =  dx, q = xmax - x0
        baixo:    p = -dy, q = y0 - ymin
        cima:     p =  dy, q = ymax - y0
    p_k < 0: a reta ENTRA pela borda k em t = q/p  -> t0 = max(t0, t)
    p_k > 0: a reta SAI pela borda k em t = q/p    -> t1 = min(t1, t)
    p_k = 0: paralela à borda; se q_k < 0 está fora.
    Se t0 > t1, não sobra nada dentro. Só calcula as interseções finais.
    """
    xmin, ymin, xmax, ymax = janela
    dx, dy = x1 - x0, y1 - y0
    bordas = (("esquerda", -dx, x0 - xmin), ("direita", dx, xmax - x0),
              ("baixo", -dy, y0 - ymin), ("cima", dy, ymax - y0))
    t0, t1 = 0.0, 1.0

    for nome, p, q in bordas:
        if p == 0:
            t = None
            if q < 0:
                if rastro is not None:
                    rastro.append((nome, p, q, t, t0, t1))
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if rastro is not None:
            rastro.append((nome, p, q, t, t0, t1))
        if t0 > t1:
            return None

    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


ALGORITMOS_RECORTE = {"Cohen–Sutherland": cohen_sutherland, "Liang–Barsky": liang_barsky}


def classificar(original, recortado, eps=1e-9):
    """'rejeitado', 'inteiro' (aceito sem mudança) ou 'recortado'."""
    if recortado is None:
        return "rejeitado"
    if all(abs(a - b) < eps for a, b in zip(original, recortado)):
        return "inteiro"
    return "recortado"


def bits(c):
    return format(c, "04b")


# ------------------------------------------------------------------ polígonos
def _vetorial(ax, ay, bx, by):
    return ax * by - ay * bx


def area_com_sinal(poligono):
    """Fórmula do laço (shoelace): positiva se os vértices estão em sentido anti-horário."""
    n = len(poligono)
    return 0.5 * sum(_vetorial(*poligono[i], *poligono[(i + 1) % n]) for i in range(n))


def retangulo_para_poligono(janela):
    xmin, ymin, xmax, ymax = janela
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def sutherland_hodgman(sujeito, janela, rastro=None):
    """Recorte de polígonos de Sutherland–Hodgman.

    sujeito: polígono qualquer (convexo ou côncavo), lista de (x, y)
    janela:  polígono CONVEXO, lista de (x, y)

    A janela é tratada como a interseção de semiplanos, um por aresta.
    O polígono passa por um "filtro" para cada aresta A->B da janela; o
    filtro percorre as arestas S->P do polígono e aplica 4 casos:
        S dentro, P dentro  -> emite P
        S dentro, P fora    -> emite a interseção I
        S fora,   P fora    -> não emite nada
        S fora,   P dentro  -> emite I e depois P
    A saída de um filtro é a entrada do próximo (pipeline).

    'dentro' = à esquerda da aresta A->B (janela em sentido anti-horário).
    """
    if area_com_sinal(janela) < 0:
        janela = janela[::-1]                       # garante anti-horário

    saida = list(sujeito)
    n = len(janela)
    for k in range(n):
        ax, ay = janela[k]
        bx, by = janela[(k + 1) % n]
        ex, ey = bx - ax, by - ay                   # direção da aresta da janela

        def dentro(pt):
            return _vetorial(ex, ey, pt[0] - ax, pt[1] - ay) >= 0

        def intersecao(s, p):
            # S + t (P - S) sobre a reta A-B:  t = -(E x (S - A)) / (E x (P - S))
            t = (-_vetorial(ex, ey, s[0] - ax, s[1] - ay)
                 / _vetorial(ex, ey, p[0] - s[0], p[1] - s[1]))
            return (s[0] + t * (p[0] - s[0]), s[1] + t * (p[1] - s[1]))

        entrada, saida = saida, []
        if not entrada:
            break
        s = entrada[-1]
        for p in entrada:
            if dentro(p):
                if not dentro(s):
                    saida.append(intersecao(s, p))
                saida.append(p)
            elif dentro(s):
                saida.append(intersecao(s, p))
            s = p
        if rastro is not None:
            rastro.append(((ax, ay), (bx, by), list(saida)))

    return saida


def pontes_degeneradas(poligono, eps=1e-9):
    """Trechos onde duas arestas do polígono se sobrepõem em sentidos opostos.

    Quando o recorte de um polígono côncavo deveria gerar pedaços separados,
    o Sutherland–Hodgman devolve um único polígono, ligando os pedaços por
    "pontes" de área zero sobre a borda da janela (ida e volta pela mesma
    linha). Retorna a lista desses trechos [((x0, y0), (x1, y1)), ...].
    """
    n = len(poligono)
    arestas = [(poligono[i], poligono[(i + 1) % n]) for i in range(n)]
    pontes = []
    for i, (a, b) in enumerate(arestas):
        d1 = (b[0] - a[0], b[1] - a[1])
        comp = (d1[0] ** 2 + d1[1] ** 2) ** 0.5
        if comp < eps:
            continue
        u = (d1[0] / comp, d1[1] / comp)               # direção unitária da aresta i
        for j in range(i + 1, n):
            c, d = arestas[j]
            d2 = (d[0] - c[0], d[1] - c[1])
            colinear = (abs(_vetorial(*d1, *d2)) < eps
                        and abs(_vetorial(*d1, c[0] - a[0], c[1] - a[1])) < eps)
            opostas = d1[0] * d2[0] + d1[1] * d2[1] < 0
            if not (colinear and opostas):
                continue
            # projeta a aresta j sobre a reta da aresta i e mede a sobreposição
            proj = lambda pt: (pt[0] - a[0]) * u[0] + (pt[1] - a[1]) * u[1]
            lo, hi = sorted((proj(c), proj(d)))
            t0, t1 = max(lo, 0.0), min(hi, comp)
            if t1 - t0 > eps:
                pontes.append(((a[0] + t0 * u[0], a[1] + t0 * u[1]),
                               (a[0] + t1 * u[0], a[1] + t1 * u[1])))
    return pontes


def exemplo_poligono():
    """Rastro do Sutherland–Hodgman (exemplo do relatório)."""
    janela = retangulo_para_poligono((10, 10, 30, 25))
    sujeito = [(5, 15), (20, 3), (35, 18), (22, 32), (18, 20)]
    print(f"Sutherland–Hodgman\n  janela  {janela}\n  sujeito {sujeito}")
    r = []
    sutherland_hodgman(sujeito, janela, r)
    for a, b, saida in r:
        pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in saida)
        print(f"  após aresta {a}->{b}: {len(saida)} vértices: {pts}")


def exemplo_rastro():
    """Mesmo segmento recortado pelos dois algoritmos (exemplo do relatório)."""
    janela = (10, 10, 30, 25)
    seg = (2, 6, 36, 30)
    print(f"Janela {janela}, segmento {seg}\n")

    print("Cohen–Sutherland")
    r = []
    res = cohen_sutherland(*seg, janela, r)
    for i, (x0, y0, c0, x1, y1, c1) in enumerate(r):
        print(f"  passo {i}: P0 = ({x0:6.2f}, {y0:6.2f}) [{bits(c0)}]   "
              f"P1 = ({x1:6.2f}, {y1:6.2f}) [{bits(c1)}]")
    print("  resultado:", tuple(round(v, 2) for v in res), "\n")

    print("Liang–Barsky")
    r = []
    res = liang_barsky(*seg, janela, r)
    print(f"  {'borda':<9} {'p':>6} {'q':>6} {'t = q/p':>9} {'t0':>7} {'t1':>7}")
    for nome, p, q, t, t0, t1 in r:
        tt = "-" if t is None else f"{t:.4f}"
        print(f"  {nome:<9} {p:>6} {q:>6} {tt:>9} {t0:>7.4f} {t1:>7.4f}")
    print("  resultado:", tuple(round(v, 2) for v in res))


def comparar(n=100_000, semente=0):
    """Concordância e tempo dos dois algoritmos em segmentos aleatórios."""
    rnd = random.Random(semente)
    janela = (-1.0, -1.0, 1.0, 1.0)
    segs = [tuple(rnd.uniform(-3, 3) for _ in range(4)) for _ in range(n)]

    resultados, tempos = {}, {}
    for nome, f in ALGORITMOS_RECORTE.items():
        inicio = time.perf_counter()
        resultados[nome] = [f(*s, janela) for s in segs]
        tempos[nome] = time.perf_counter() - inicio

    cs, lb = resultados.values()
    iguais = sum((a is None and b is None) or
                 (a is not None and b is not None and
                  all(abs(u - v) < 1e-9 for u, v in zip(a, b)))
                 for a, b in zip(cs, lb))
    contagem = {}
    for s, r in zip(segs, cs):
        k = classificar(s, r)
        contagem[k] = contagem.get(k, 0) + 1

    print(f"\n{n} segmentos aleatórios em [-3, 3]^2, janela [-1, 1]^2")
    print(f"  classificação: {contagem}")
    print(f"  resultados idênticos: {iguais}/{n}")
    for nome, t in tempos.items():
        print(f"  {nome:<17} {t * 1000:8.1f} ms  ({t / n * 1e6:.2f} µs por segmento)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "poligono":
        exemplo_poligono()
    else:
        exemplo_rastro()
        comparar()
