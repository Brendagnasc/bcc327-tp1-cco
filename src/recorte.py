"""
recorte.py

Algoritmos clássicos de recorte (clipping) implementados pelo grupo.

Segmentos de reta contra uma janela retangular alinhada aos eixos
janela = (xmin, ymin, xmax, ymax):
- cohen_sutherland: códigos de região de 4 bits + recorte iterativo
- liang_barsky: forma paramétrica P(t) = P0 + t (P1 - P0), t em [0, 1]

Cada função retorna o segmento recortado (x0, y0, x1, y1) ou None se
o segmento estiver totalmente fora.

Polígonos:
- sutherland_hodgman: janela convexa; recorta o sujeito borda a borda
- weiler_atherton: janela e sujeito quaisquer (côncavos, sem buracos);
  percorre as duas fronteiras e devolve uma LISTA de polígonos
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


def ponto_dentro(x, y, poligono):
    """Teste par-ímpar: conta quantas arestas um raio horizontal para a direita cruza."""
    dentro = False
    n = len(poligono)
    for i in range(n):
        (x0, y0), (x1, y1) = poligono[i], poligono[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            dentro = not dentro
    return dentro


class _No:
    """Vértice das listas do Weiler–Atherton (original ou interseção)."""
    __slots__ = ("x", "y", "intersecao", "entrada", "par", "visitado")

    def __init__(self, x, y, intersecao=False):
        self.x, self.y = x, y
        self.intersecao = intersecao
        self.entrada = False
        self.par = None          # o mesmo ponto na outra lista
        self.visitado = False


def _ponto_na_aresta(p, a, b, eps):
    ex, ey = b[0] - a[0], b[1] - a[1]
    comp2 = ex * ex + ey * ey
    if comp2 == 0:
        return False
    t = ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / comp2
    if t < -eps or t > 1 + eps:
        return False
    return abs(_vetorial(ex, ey, p[0] - a[0], p[1] - a[1])) / comp2 ** 0.5 < eps


def _degenerado(sujeito, janela, eps):
    """Há vértice de um polígono sobre aresta do outro (inclui arestas sobrepostas)?"""
    for p, q in ((sujeito, janela), (janela, sujeito)):
        m = len(q)
        for v in p:
            if any(_ponto_na_aresta(v, q[k], q[(k + 1) % m], eps) for k in range(m)):
                return True
    return False


def weiler_atherton(sujeito, janela, rastro=None):
    """Recorte de polígonos de Weiler–Atherton (interseção sujeito ∩ janela).

    1. Os dois polígonos são colocados em sentido anti-horário.
    2. Calculam-se todas as interseções entre arestas do sujeito e da janela;
       cada uma é inserida nas DUAS listas de vértices, na posição certa, e
       as duas cópias são ligadas (par).
    3. Percorrendo o sujeito, cada interseção alterna entre "entrada" (o
       sujeito entra na janela) e "saída", a partir de o primeiro vértice
       estar dentro ou fora.
    4. Para cada entrada não visitada: segue o SUJEITO até a próxima
       interseção (uma saída), troca para a JANELA e segue até a próxima
       interseção (uma entrada), troca de novo... até voltar ao início.
       Cada volta fechada é um polígono do resultado.

    Casos degenerados (vértice exatamente sobre aresta, comuns em grade
    inteira) são evitados deslocando a janela por um valor desprezível.
    """
    if len(sujeito) < 3 or len(janela) < 3:
        return []
    if area_com_sinal(sujeito) < 0:
        sujeito = sujeito[::-1]
    if area_com_sinal(janela) < 0:
        janela = janela[::-1]

    xs = [p[0] for p in sujeito + janela]
    ys = [p[1] for p in sujeito + janela]
    escala = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    eps = 1e-9 * escala
    for tentativa in range(1, 6):
        if not _degenerado(sujeito, janela, eps):
            break
        d = 1e-6 * escala * tentativa
        janela = [(x + d, y + 0.618 * d) for x, y in janela]

    # --- 2. listas com as interseções inseridas
    lista_s = [[_No(*p)] for p in sujeito]     # por aresta: vértice + interseções
    lista_j = [[_No(*p)] for p in janela]
    extras_s = [[] for _ in sujeito]
    extras_j = [[] for _ in janela]
    ns, nj = len(sujeito), len(janela)
    for i in range(ns):
        a, b = sujeito[i], sujeito[(i + 1) % ns]
        r = (b[0] - a[0], b[1] - a[1])
        for k in range(nj):
            c, d = janela[k], janela[(k + 1) % nj]
            s_ = (d[0] - c[0], d[1] - c[1])
            den = _vetorial(*r, *s_)
            if den == 0:
                continue                                    # paralelas
            ac = (c[0] - a[0], c[1] - a[1])
            t = _vetorial(*ac, *s_) / den                   # parâmetro na aresta do sujeito
            u = _vetorial(*ac, *r) / den                    # parâmetro na aresta da janela
            if 0 < t < 1 and 0 < u < 1:
                x, y = a[0] + t * r[0], a[1] + t * r[1]
                no_s, no_j = _No(x, y, True), _No(x, y, True)
                no_s.par, no_j.par = no_j, no_s
                extras_s[i].append((t, no_s))
                extras_j[k].append((u, no_j))

    for blocos, extras in ((lista_s, extras_s), (lista_j, extras_j)):
        for bloco, ex in zip(blocos, extras):
            bloco.extend(no for _, no in sorted(ex, key=lambda e: e[0]))
    S = [no for bloco in lista_s for no in bloco]
    J = [no for bloco in lista_j for no in bloco]

    # --- sem interseções: um contém o outro, ou são disjuntos
    if not any(no.intersecao for no in S):
        if ponto_dentro(*sujeito[0], janela):
            return [list(sujeito)]
        if ponto_dentro(*janela[0], sujeito):
            return [list(janela)]
        return []

    # --- 3. entrada / saída
    dentro = ponto_dentro(S[0].x, S[0].y, janela)
    for no in S:
        if no.intersecao:
            no.entrada = not dentro
            dentro = not dentro
    if rastro is not None:
        rastro.extend((no.x, no.y, "entrada" if no.entrada else "saída")
                      for no in S if no.intersecao)

    # --- 4. percurso
    pos = {id(no): i for i, no in enumerate(S)}
    pos.update({id(no): i for i, no in enumerate(J)})
    resultado = []
    for inicio in S:
        if not (inicio.intersecao and inicio.entrada and not inicio.visitado):
            continue
        poligono = []
        lista, idx = S, pos[id(inicio)]
        while True:
            no = lista[idx]
            no.visitado = no.par.visitado = True
            poligono.append((no.x, no.y))
            idx = (idx + 1) % len(lista)
            while not lista[idx].intersecao:                # vértices até a próxima interseção
                poligono.append((lista[idx].x, lista[idx].y))
                idx = (idx + 1) % len(lista)
            chegada = lista[idx]
            if chegada is inicio or chegada.par is inicio:
                break
            lista = J if lista is S else S                  # troca de fronteira
            idx = pos[id(chegada.par)]
        resultado.append(poligono)
    return resultado


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

    # Polígono em U: o resultado correto são dois pedaços
    janela = retangulo_para_poligono((0, 12, 30, 20))
    sujeito = [(2, 0), (28, 0), (28, 24), (20, 24), (20, 8), (10, 8), (10, 24), (2, 24)]
    print(f"\nPolígono em U contra a janela {janela}")
    sh = sutherland_hodgman(sujeito, janela)
    print(f"  Sutherland–Hodgman: 1 polígono com {len(sh)} vértices, "
          f"pontes degeneradas: {pontes_degeneradas(sh)}")
    r = []
    wa = weiler_atherton(sujeito, janela, r)
    print("  Weiler–Atherton: interseções ao longo do sujeito:")
    for x, y, tipo in r:
        print(f"    ({x:.1f}, {y:.1f}) {tipo}")
    for i, p in enumerate(wa):
        pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in p)
        print(f"  polígono {i + 1}: {pts}")


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
