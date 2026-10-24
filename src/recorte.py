"""
recorte.py

Algoritmos clássicos de recorte (clipping) implementados pelo grupo.

Segmentos de reta contra uma janela retangular alinhada aos eixos
janela = (xmin, ymin, xmax, ymax):
- cohen_sutherland: códigos de região de 4 bits + recorte iterativo
- liang_barsky: forma paramétrica P(t) = P0 + t (P1 - P0), t em [0, 1]

Cada função retorna o segmento recortado (x0, y0, x1, y1) ou None se
o segmento estiver totalmente fora.
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
    exemplo_rastro()
    comparar()
