"""
raycasting.py

Ray Casting implementado pelo grupo para selecionar vasos com o mouse.

1. Raio a partir da câmera:  P(t) = O + t D.
   O pixel clicado vira coordenadas normalizadas (NDC) x, y em [-1, 1].
   Os pontos (x, y, -1) e (x, y, +1) do cubo NDC estão nos planos near e
   far; aplicando a inversa de P . V (e dividindo por w) eles voltam ao
   espaço do mundo. O = ponto no near, D = far - near. Funciona para as
   duas projeções: na ortográfica os raios saem paralelos.
2. O raio é levado ao espaço do MODELO com a inversa de M (D com w = 0),
   onde estão os pontos e os raios dos vasos. O parâmetro t é o mesmo nos
   dois espaços, então as distâncias continuam comparáveis.
3. Interseção com cada vaso, modelado como um cilindro de raio r entre A e B
   mais uma esfera de raio r em cada ponta (a mesma forma dos tubos).
4. O vaso escolhido é o de MENOR t positivo: o primeiro que o raio encontra.
"""

import numpy as np


def raio_do_pixel(px, py, largura, altura, projecao, visao):
    """Origem e direção (espaço do mundo) do raio que passa pelo pixel (px, py)."""
    x = 2.0 * (px + 0.5) / largura - 1.0
    y = 1.0 - 2.0 * (py + 0.5) / altura            # a tela cresce para baixo; o NDC, para cima
    inversa = np.linalg.inv(projecao @ visao)
    perto = inversa @ np.array([x, y, -1.0, 1.0])
    longe = inversa @ np.array([x, y, 1.0, 1.0])
    perto, longe = perto[:3] / perto[3], longe[:3] / longe[3]
    return perto, longe - perto


def para_modelo(origem, direcao, modelo):
    """Leva o raio para o espaço do modelo (ponto com w = 1, direção com w = 0)."""
    inv = np.linalg.inv(modelo)
    return (inv @ np.append(origem, 1.0))[:3], inv[:3, :3] @ direcao


def intersecao_cilindros(O, D, A, B, r):
    """Menor t > 0 de interseção do raio com cada cilindro finito AB de raio r.

    Decompondo em relação ao eixo unitário a = (B - A)/|B - A|, com m = O - A,
    um ponto está na superfície lateral se a parte perpendicular ao eixo tem
    comprimento r:  |(m + tD) - ((m + tD) . a) a|^2 = r^2, que dá
        k2 t^2 + k1 t + k0 = 0
        k2 = D.D - (D.a)^2
        k1 = 2 (m.D - (m.a)(D.a))
        k0 = m.m - (m.a)^2 - r^2
    A raiz só vale se a projeção do ponto no eixo, s = (m + tD) . a, estiver
    entre 0 e |B - A| (dentro do comprimento do vaso). Retorna inf se não há.
    """
    eixo = B - A
    comp = np.linalg.norm(eixo, axis=1)
    a = eixo / comp[:, None]
    m = O - A
    Da = a @ D
    ma = (m * a).sum(axis=1)
    k2 = D @ D - Da ** 2
    k1 = 2 * (m @ D - ma * Da)
    k0 = (m * m).sum(axis=1) - ma ** 2 - r ** 2

    disc = k1 ** 2 - 4 * k2 * k0
    t = np.full(len(A), np.inf)
    ok = (disc >= 0) & (k2 > 1e-18)
    raiz = np.sqrt(np.where(ok, disc, 0.0))
    for sinal in (-1.0, 1.0):                      # primeiro a entrada, depois a saída
        tc = (-k1 + sinal * raiz) / np.where(ok, 2 * k2, 1.0)
        s = ma + tc * Da
        valido = ok & (tc > 0) & (s >= 0) & (s <= comp) & np.isinf(t)
        t = np.where(valido, tc, t)
    return t


def intersecao_esferas(O, D, C, r):
    """Menor t > 0 de interseção do raio com esferas de centro C e raio r:
    |O + tD - C|^2 = r^2  ->  (D.D) t^2 + 2 (m.D) t + (m.m - r^2) = 0, m = O - C."""
    m = O - C
    k2 = D @ D
    k1 = 2 * (m @ D)
    k0 = (m * m).sum(axis=1) - r ** 2
    disc = k1 ** 2 - 4 * k2 * k0
    ok = disc >= 0
    raiz = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-k1 - raiz) / (2 * k2)
    t2 = (-k1 + raiz) / (2 * k2)
    t = np.where(ok & (t1 > 0), t1, np.where(ok & (t2 > 0), t2, np.inf))
    return t


def escolher_vaso(O, D, pontos, segmentos, raio):
    """Índice do primeiro vaso atingido pelo raio e o t da interseção (ou None)."""
    A, B = pontos[segmentos[:, 0]], pontos[segmentos[:, 1]]
    t = np.minimum(intersecao_cilindros(O, D, A, B, raio),
                   np.minimum(intersecao_esferas(O, D, A, raio),
                              intersecao_esferas(O, D, B, raio)))
    i = int(np.argmin(t))
    return (None, None) if np.isinf(t[i]) else (i, float(t[i]))


if __name__ == "__main__":
    # exemplo numérico: cilindro de (0,0,0) a (0,0,10), raio 1; raio de luz
    # saindo de (5,0,5) na direção -x: deve entrar no cilindro em (1,0,5), t = 4
    A = np.array([[0.0, 0.0, 0.0]])
    B = np.array([[0.0, 0.0, 10.0]])
    O, D = np.array([5.0, 0.0, 5.0]), np.array([-1.0, 0.0, 0.0])
    t = intersecao_cilindros(O, D, A, B, np.array([1.0]))[0]
    print(f"raio O = {O}, D = {D}; cilindro A = {A[0]}, B = {B[0]}, r = 1")
    print(f"  t = {t:.2f}, ponto = {O + t * D}")
    # mesmo raio deslocado para z = 12: passa acima do cilindro, só acertaria a "tampa"
    O2 = np.array([5.0, 0.0, 12.0])
    print(f"raio deslocado para z = 12: t = {intersecao_cilindros(O2, D, A, B, np.array([1.0]))[0]}")
