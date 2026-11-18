"""
raytracer.py

Mini Ray Tracer implementado pelo grupo em NumPy (sem OpenGL).

Cena simplificada inspirada nas árvores CCO: uma bifurcação de vasos
(cilindros e esferas nas junções) sobre um piso quadriculado, uma esfera
espelhada e uma esfera fosca, iluminadas por uma luz pontual.

Para cada pixel:
  1. raio PRIMÁRIO da câmera:  P(t) = O + t D
  2. interseção mais próxima com todos os objetos (menor t > 0)
  3. iluminação local de Blinn-Phong no ponto atingido
  4. raio de SOMBRA do ponto até a luz: se algum objeto estiver no
     caminho, só a componente ambiente permanece
  5. raio REFLETIDO  R = D - 2 (D . N) N  em superfícies espelhadas,
     traçado recursivamente até a profundidade máxima

Os três modos mostram a diferença entre as técnicas:
  1 = Ray Casting (só passos 1 a 3)
  2 = + sombras (passo 4)
  3 = Ray Tracing completo (+ reflexões, passo 5)

Uso:
    python src/raytracer.py [--largura 640]
Teclas: 1 2 3 modo | S salva em report/figures/ | ESC sai
"""

import sys
import time
from pathlib import Path

import numpy as np

from cores import mapear_cores
from transformacoes import olhar_para

EPS = 1e-4
LUZ = np.array([2.5, 4.0, 3.0])
CEU_BAIXO, CEU_ALTO = np.array([0.95, 0.96, 1.0]), np.array([0.55, 0.70, 0.95])
KA, KD, KS = 0.15, 0.8, 0.5
MODOS = {1: ("Ray Casting", False, 0), 2: ("Ray Casting + sombras", True, 0),
         3: ("Ray Tracing (sombras + reflexões)", True, 3)}


# ------------------------------------------------------------------ cena
def montar_cena():
    """Lista de objetos: (tipo, geometria, material)."""
    # bifurcação: vaso pai e dois filhos, raios decrescentes como numa árvore CCO
    vasos = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.14),
             ((0.0, 1.0, 0.0), (-0.55, 1.75, 0.15), 0.10),
             ((0.0, 1.0, 0.0), (0.50, 1.70, -0.20), 0.08)]
    raios = np.array([r for _, _, r in vasos])
    cores = mapear_cores(raios, 0.06, 0.14, "jet")
    cena = []
    for (a, b, r), cor in zip(vasos, cores):
        mat = {"cor": cor, "brilho": 40, "reflexo": 0.0}
        cena.append(("cilindro", (np.array(a), np.array(b), r), mat))
        cena.append(("esfera", (np.array(b), r), mat))     # junção / ponta arredondada
    cena.append(("esfera", (np.array([1.15, 0.45, 0.55]), 0.45),
                 {"cor": np.array([0.9, 0.9, 0.95]), "brilho": 120, "reflexo": 0.8}))
    cena.append(("esfera", (np.array([-1.05, 0.32, 0.75]), 0.32),
                 {"cor": np.array([0.2, 0.45, 0.85]), "brilho": 20, "reflexo": 0.0}))
    cena.append(("plano", None, {"cor": None, "brilho": 10, "reflexo": 0.25}))
    return cena


# ------------------------------------------------------------------ interseções
# Todas recebem N raios (O, D com D unitário) e devolvem t (N,), inf se não há.
def _menor_positiva(b, disc):
    raiz = np.sqrt(np.where(disc >= 0, disc, 0.0))
    t1, t2 = -b - raiz, -b + raiz
    t = np.where(t1 > EPS, t1, np.where(t2 > EPS, t2, np.inf))
    return np.where(disc >= 0, t, np.inf)


def t_esfera(O, D, centro, r):
    # |O + tD - C|^2 = r^2  com |D| = 1:  t^2 + 2 b t + c = 0
    m = O - centro
    b = (m * D).sum(axis=1)
    c = (m * m).sum(axis=1) - r * r
    return _menor_positiva(b, b * b - c)


def t_plano(O, D):
    # piso y = 0:  O_y + t D_y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -O[:, 1] / D[:, 1]
    return np.where((t > EPS) & np.isfinite(t), t, np.inf)


def t_cilindro(O, D, A, B, r):
    # parte perpendicular ao eixo com comprimento r (mesma conta do raycasting.py)
    eixo = B - A
    comp = np.linalg.norm(eixo)
    a = eixo / comp
    m = O - A
    Da, ma = D @ a, m @ a
    k2 = 1.0 - Da ** 2
    k1 = (m * D).sum(axis=1) - ma * Da           # metade do coeficiente linear
    k0 = (m * m).sum(axis=1) - ma ** 2 - r * r
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = k1 ** 2 - k2 * k0
        raiz = np.sqrt(np.where(disc >= 0, disc, 0.0))
        t = np.full(len(O), np.inf)
        for sinal in (-1.0, 1.0):
            tc = (-k1 + sinal * raiz) / k2
            s = ma + tc * Da
            ok = (disc >= 0) & (k2 > 1e-12) & (tc > EPS) & (s >= 0) & (s <= comp) & np.isinf(t)
            t = np.where(ok, tc, t)
    return t


def normal_em(tipo, geo, P):
    if tipo == "esfera":
        centro, r = geo
        return (P - centro) / r
    if tipo == "plano":
        return np.tile([0.0, 1.0, 0.0], (len(P), 1))
    A, B, r = geo                                  # cilindro: tira a parte axial
    a = (B - A) / np.linalg.norm(B - A)
    v = P - A
    n = v - (v @ a)[:, None] * a
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def intersecao_mais_proxima(cena, O, D):
    melhor_t = np.full(len(O), np.inf)
    melhor_obj = np.full(len(O), -1)
    for k, (tipo, geo, _) in enumerate(cena):
        if tipo == "esfera":
            t = t_esfera(O, D, *geo)
        elif tipo == "plano":
            t = t_plano(O, D)
        else:
            t = t_cilindro(O, D, *geo)
        mais_perto = t < melhor_t
        melhor_t[mais_perto], melhor_obj[mais_perto] = t[mais_perto], k
    return melhor_t, melhor_obj


# ------------------------------------------------------------------ traçado
def tracar(cena, O, D, sombras, profundidade, stats):
    """Cor de cada raio (N, 3). Recursivo nas reflexões."""
    t, obj = intersecao_mais_proxima(cena, O, D)
    cor = CEU_BAIXO + (CEU_ALTO - CEU_BAIXO) * np.clip(D[:, 1:2], 0, 1)   # céu
    acertou = np.isfinite(t)

    for k, (tipo, geo, mat) in enumerate(cena):
        sel = acertou & (obj == k)
        if not sel.any():
            continue
        o, d = O[sel], D[sel]
        P = o + t[sel, None] * d
        N = normal_em(tipo, geo, P)
        N = np.where(((N * d).sum(axis=1) > 0)[:, None], -N, N)    # normal contra o raio

        if mat["cor"] is None:                     # piso quadriculado
            casa = (np.floor(P[:, 0] * 2) + np.floor(P[:, 2] * 2)) % 2
            base = np.where(casa[:, None] == 0, [0.85, 0.85, 0.85], [0.35, 0.38, 0.45])
        else:
            base = np.tile(mat["cor"], (len(P), 1))

        para_luz = LUZ - P
        dist_luz = np.linalg.norm(para_luz, axis=1)
        L = para_luz / dist_luz[:, None]
        iluminado = np.ones(len(P), dtype=bool)
        if sombras:
            # raio de sombra: sai um pouco acima da superfície para não se auto-atingir
            ts, _ = intersecao_mais_proxima(cena, P + EPS * 10 * N, L)
            iluminado = ts >= dist_luz
            stats["sombra"] += len(P)

        V = -d
        H = L + V
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        difusa = np.maximum((N * L).sum(axis=1), 0.0)
        especular = np.maximum((N * H).sum(axis=1), 0.0) ** mat["brilho"] * (difusa > 0)
        local = KA * base + iluminado[:, None] * (KD * base * difusa[:, None]
                                                  + KS * especular[:, None])

        kr = mat["reflexo"]
        if profundidade > 0 and kr > 0:
            R = d - 2 * (d * N).sum(axis=1, keepdims=True) * N
            stats["reflexo"] += len(P)
            refletida = tracar(cena, P + EPS * 10 * N, R, sombras, profundidade - 1, stats)
            local = (1 - kr) * local + kr * refletida
        cor[sel] = local
    return np.clip(cor, 0.0, 1.0)


def renderizar(modo, largura=640, altura=480):
    """Imagem (altura, largura, 3) uint8 e estatísticas de raios."""
    _, sombras, profundidade = MODOS[modo]
    cena = montar_cena()
    olho, alvo = np.array([0.3, 1.5, 4.3]), np.array([0.0, 0.8, 0.0])
    V = olhar_para(olho, alvo, (0, 1, 0))
    rot_mundo = V[:3, :3].T                          # câmera -> mundo (inversa da rotação)

    # direção de cada pixel no espaço da câmera (plano de imagem em z = -1)
    meia = np.tan(np.radians(45.0) / 2)
    i, j = np.meshgrid(np.arange(largura), np.arange(altura))
    x = (2 * (i + 0.5) / largura - 1) * meia * largura / altura
    y = (1 - 2 * (j + 0.5) / altura) * meia
    d_cam = np.stack([x, y, -np.ones_like(x)], axis=-1).reshape(-1, 3)
    D = d_cam @ rot_mundo.T
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    O = np.tile(olho, (len(D), 1))

    stats = {"primarios": len(D), "sombra": 0, "reflexo": 0}
    inicio = time.perf_counter()
    cor = tracar(cena, O, D, sombras, profundidade, stats)
    stats["segundos"] = time.perf_counter() - inicio
    return (cor.reshape(altura, largura, 3) * 255).astype(np.uint8), stats


def main():
    import pygame
    largura = int(sys.argv[sys.argv.index("--largura") + 1]) if "--largura" in sys.argv else 640
    altura = largura * 3 // 4
    imagens = {}
    for modo, (nome, _, _) in MODOS.items():
        img, st = renderizar(modo, largura, altura)
        imagens[modo] = (img, st)
        print(f"{nome}: {st['primarios']} primários, {st['sombra']} de sombra, "
              f"{st['reflexo']} refletidos, {st['segundos']:.2f} s")

    pygame.display.init()
    pygame.font.init()
    tela = pygame.display.set_mode((largura, altura + 30))
    fonte = pygame.font.Font(None, 22)
    modo = 3
    rodando = True
    while rodando:
        img, st = imagens[modo]
        tela.fill((255, 255, 255))
        tela.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        legenda = (f"{modo}: {MODOS[modo][0]} | raios: {st['primarios']} primários, "
                   f"{st['sombra']} sombra, {st['reflexo']} reflexão | 1 2 3  S salvar  ESC")
        tela.blit(fonte.render(legenda, True, (20, 20, 20)), (8, altura + 7))
        pygame.display.set_caption(f"Mini Ray Tracer - {MODOS[modo][0]}")
        pygame.display.flip()
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN
                                              and evento.key == pygame.K_ESCAPE):
                rodando = False
            elif evento.type == pygame.KEYDOWN and evento.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                modo = evento.key - pygame.K_0
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_s:
                pasta = Path("report/figures")
                pasta.mkdir(parents=True, exist_ok=True)
                caminho = pasta / f"rt_modo{modo}.png"
                pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), str(caminho))
                print("salvo:", caminho)
        pygame.time.wait(30)
    pygame.quit()


if __name__ == "__main__":
    main()
