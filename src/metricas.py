"""
metricas.py

Métricas geométricas e topológicas da árvore CCO, calculadas pelo grupo.

A árvore é tratada como um grafo: pontos são nós e segmentos são arestas.
    grau 1 -> extremidade (a raiz ou um terminal)
    grau 3 -> bifurcação
A raiz é a extremidade cujo segmento tem o maior raio. Uma busca em largura
(BFS) a partir dela orienta os segmentos de pai para filho, o que permite
calcular a GERAÇÃO de cada vaso (quantas bifurcações há entre ele e a raiz)
e comparar os raios de pai e filhos em cada bifurcação.

Uso:
    python src/metricas.py data/tree3D_*.vtk
"""

import sys
from collections import deque

import numpy as np

MM = 1e3        # coordenadas em m -> mm (o raio já está em mm)


def analisar(arvore):
    pts, seg, r = arvore.pontos, arvore.segmentos, arvore.raios
    n = len(pts)
    grau = np.bincount(seg.ravel(), minlength=n)

    # lista de adjacência: para cada ponto, (vizinho, índice do segmento)
    vizinhos = [[] for _ in range(n)]
    for k, (a, b) in enumerate(seg):
        vizinhos[a].append((b, k))
        vizinhos[b].append((a, k))

    extremidades = np.nonzero(grau == 1)[0]
    seg_da_ext = np.array([vizinhos[p][0][1] for p in extremidades])
    raiz = int(extremidades[np.argmax(r[seg_da_ext])])

    # BFS: geração de cada segmento e segmento pai
    geracao = np.full(len(seg), -1)
    pai = np.full(len(seg), -1)
    fila = deque([(raiz, -1, -1)])          # (ponto, segmento de chegada, geração dele)
    visitado = np.zeros(n, dtype=bool)
    visitado[raiz] = True
    while fila:
        p, s_chegada, g = fila.popleft()
        for q, k in vizinhos[p]:
            if not visitado[q]:
                visitado[q] = True
                geracao[k] = g + 1
                pai[k] = s_chegada
                fila.append((q, k, g + 1))

    comprimento = np.linalg.norm(pts[seg[:, 1]] - pts[seg[:, 0]], axis=1) * MM   # mm
    volume = np.pi * r ** 2 * comprimento                                         # mm^3

    # bifurcações: razão r_pai^3 / (r_filho1^3 + r_filho2^3)  (1 = lei de Murray)
    filhos = [[] for _ in range(len(seg))]
    for k, p_ in enumerate(pai):
        if p_ >= 0:
            filhos[p_].append(k)
    razoes = [r[k] ** 3 / sum(r[f] ** 3 for f in fs) for k, fs in enumerate(filhos) if len(fs) == 2]

    return {
        "raiz": raiz,
        "geracao": geracao,
        "comprimento": comprimento,
        "volume": volume,
        "n_bifurcacoes": int((grau == 3).sum()),
        "n_terminais": int(len(extremidades) - 1),
        "geracao_max": int(geracao.max()),
        "comprimento_total": float(comprimento.sum()),
        "volume_total": float(volume.sum()),
        "murray_media": float(np.mean(razoes)) if razoes else float("nan"),
        "murray_desvio": float(np.std(razoes)) if razoes else float("nan"),
    }


def linhas_painel(arvore, met, visiveis=None):
    """Texto do painel do visualizador; 'visiveis' = máscara do filtro por raio."""
    r = arvore.raios
    linhas = [
        f"vasos: {arvore.n_segmentos}  |  bifurcações: {met['n_bifurcacoes']}  |  "
        f"terminais: {met['n_terminais']}",
        f"gerações: 0 a {met['geracao_max']}  |  raio: {r.min():.3f} a {r.max():.3f} mm "
        f"(média {r.mean():.3f})",
        f"comprimento total: {met['comprimento_total']:.1f} mm  |  "
        f"volume total: {met['volume_total']:.1f} mm³",
        f"r_pai³ / (r_f1³ + r_f2³): {met['murray_media']:.3f} ± {met['murray_desvio']:.3f}",
    ]
    if visiveis is not None and not visiveis.all():
        linhas.append(f"no filtro: {int(visiveis.sum())} vasos  |  "
                      f"comprimento {met['comprimento'][visiveis].sum():.1f} mm  |  "
                      f"volume {met['volume'][visiveis].sum():.1f} mm³")
    return [l.replace(".", ",") for l in linhas]


if __name__ == "__main__":
    from leitor_vtk import ler_vtk
    if len(sys.argv) < 2:
        print("uso: python src/metricas.py <arquivo.vtk> [...]")
        sys.exit(1)
    print(f"{'modelo':<17} {'bif.':>5} {'term.':>5} {'ger.':>4} {'compr. (mm)':>12} "
          f"{'vol. (mm³)':>11} {'r_p³/Σr_f³':>12}")
    for arq in sys.argv[1:]:
        arv = ler_vtk(arq)
        m = analisar(arv)
        print(f"{arv.nome:<17} {m['n_bifurcacoes']:>5} {m['n_terminais']:>5} "
              f"{m['geracao_max']:>4} {m['comprimento_total']:>12.1f} {m['volume_total']:>11.1f} "
              f"{m['murray_media']:>7.3f} ± {m['murray_desvio']:.3f}")
