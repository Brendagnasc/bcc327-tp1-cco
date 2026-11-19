"""
raytracer_arvore.py

Ray Tracer da árvore CCO de verdade (lida do VTK), em NumPy, com sombras e
reflexo no piso. A imagem é calculada aos poucos, em faixas horizontais, e
a janela mostra o progresso.

Diferença para o raytracer.py: lá a cena tinha 9 objetos; aqui são mais de
mil (um cilindro por vaso e uma esfera por ponto). Testar cada raio contra
cada objeto custaria centenas de milhões de testes. A solução usada é uma
etapa de PODA com esferas envolventes:

  1. cada objeto ganha uma esfera que o contém (centro no meio do vaso,
     raio = metade do comprimento + raio do vaso);
  2. para um bloco de raios, o teste raio-esfera envolvente é feito contra
     TODOS os objetos de uma vez com produtos de matrizes (barato);
  3. só os pares (raio, objeto) que passaram na poda recebem o teste exato
     raio-cilindro ou raio-esfera, que é mais caro.

Uso:
    python src/raytracer_arvore.py [arquivo.vtk] [--largura 640] [--rapido]
    (o visualizador abre este programa com a tecla Y, passando a vista atual)
Teclas na janela: S salva em report/figures/ | ESC sai
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cores import mapear_cores
from leitor_vtk import ler_vtk
from textura import PLANO_Y

EPS = 1e-4
BLOCO = 2048                  # raios testados juntos na poda (memória: BLOCO x objetos)
KA, KD, KS, BRILHO = 0.18, 0.78, 0.45, 48.0
REFLEXO_PISO = 0.45


# ------------------------------------------------------------------ cena
class CenaArvore:
    def __init__(self, pontos, segmentos, raios, cores_seg):
        """pontos já no espaço do mundo; raios já na escala do mundo."""
        A, B = pontos[segmentos[:, 0]], pontos[segmentos[:, 1]]
        self.A, self.B, self.r = A, B, raios
        eixo = B - A
        self.comp = np.linalg.norm(eixo, axis=1)
        self.a = eixo / self.comp[:, None]

        # esferas nos pontos: raio e cor do maior vaso que toca o ponto
        ordem = np.argsort(raios)
        seg_do_ponto = np.full(len(pontos), -1)
        seg_do_ponto[segmentos[ordem, 0]] = ordem
        seg_do_ponto[segmentos[ordem, 1]] = ordem
        usados = np.nonzero(seg_do_ponto >= 0)[0]
        self.C, self.rc = pontos[usados], raios[seg_do_ponto[usados]]

        self.m = len(A)                          # objetos 0..m-1: cilindros; m..: esferas
        self.cor = np.vstack([cores_seg, cores_seg[seg_do_ponto[usados]]])
        # esferas envolventes de todos os objetos
        self.env_c = np.vstack([(A + B) / 2, self.C]).astype(np.float32)
        self.env_r = np.concatenate([self.comp / 2 + raios, self.rc]).astype(np.float32)
        self.env_c2 = (self.env_c.astype(np.float64) ** 2).sum(axis=1)

    # -------------------------------------------------------------- interseção
    def intersecao(self, O, D):
        """Menor t > 0 e índice do objeto atingido por cada raio (inf, -1 se nenhum)."""
        n = len(O)
        t_melhor = np.full(n, np.inf)
        obj = np.full(n, -1)
        for ini in range(0, n, BLOCO):
            o, d = O[ini:ini + BLOCO], D[ini:ini + BLOCO]
            # 1. poda: raio x esfera envolvente, para todos os objetos ao mesmo tempo
            #    tca = (C - O) . D ;  d2 = |C - O|^2 - tca^2  (com |D| = 1)
            o32, d32 = o.astype(np.float32), d.astype(np.float32)
            tca = d32 @ self.env_c.T - (o32 * d32).sum(axis=1, keepdims=True)
            dist2 = (self.env_c2[None, :] - 2 * (o32 @ self.env_c.T)
                     + (o32 * o32).sum(axis=1, keepdims=True))
            candidato = ((dist2 - tca * tca) <= self.env_r ** 2) & (tca + self.env_r > EPS)
            ri, oi = np.nonzero(candidato)
            if len(ri) == 0:
                continue
            # 2. teste exato só nos pares que passaram
            t = np.full(len(ri), np.inf)
            cil = oi < self.m
            if cil.any():
                t[cil] = self._t_cilindros(o[ri[cil]], d[ri[cil]], oi[cil])
            esf = ~cil
            if esf.any():
                t[esf] = self._t_esferas(o[ri[esf]], d[ri[esf]], oi[esf] - self.m)
            # 3. menor t de cada raio
            tb = np.full(len(o), np.inf)
            np.minimum.at(tb, ri, t)
            venceu = np.isfinite(t) & (t == tb[ri])
            ob = np.full(len(o), -1)
            ob[ri[venceu]] = oi[venceu]
            t_melhor[ini:ini + BLOCO], obj[ini:ini + BLOCO] = tb, ob
        return t_melhor, obj

    def _t_cilindros(self, O, D, k):
        A, a, r, comp = self.A[k], self.a[k], self.r[k], self.comp[k]
        m = O - A
        Da, ma = (D * a).sum(axis=1), (m * a).sum(axis=1)
        k2 = 1.0 - Da ** 2
        k1 = (m * D).sum(axis=1) - ma * Da            # metade do termo linear
        k0 = (m * m).sum(axis=1) - ma ** 2 - r * r
        disc = k1 ** 2 - k2 * k0
        ok = (disc >= 0) & (k2 > 1e-12)
        raiz = np.sqrt(np.where(ok, disc, 0.0))
        den = np.where(ok, k2, 1.0)
        t = np.full(len(O), np.inf)
        for sinal in (-1.0, 1.0):
            tc = (-k1 + sinal * raiz) / den
            s = ma + tc * Da
            valido = ok & (tc > EPS) & (s >= 0) & (s <= comp) & np.isinf(t)
            t = np.where(valido, tc, t)
        return t

    def _t_esferas(self, O, D, k):
        m = O - self.C[k]
        b = (m * D).sum(axis=1)
        c = (m * m).sum(axis=1) - self.rc[k] ** 2
        disc = b * b - c
        raiz = np.sqrt(np.where(disc >= 0, disc, 0.0))
        t1, t2 = -b - raiz, -b + raiz
        t = np.where(t1 > EPS, t1, np.where(t2 > EPS, t2, np.inf))
        return np.where(disc >= 0, t, np.inf)

    def normal(self, P, obj):
        N = np.zeros_like(P)
        cil = obj < self.m
        if cil.any():
            k = obj[cil]
            v = P[cil] - self.A[k]
            n = v - (v * self.a[k]).sum(axis=1, keepdims=True) * self.a[k]
            N[cil] = n
        esf = ~cil
        if esf.any():
            N[esf] = P[esf] - self.C[obj[esf] - self.m]
        return N / np.linalg.norm(N, axis=1, keepdims=True)


# ------------------------------------------------------------------ traçado
class Tracador:
    def __init__(self, cena, luz, fundo_baixo, fundo_alto, escuro):
        self.cena, self.luz = cena, np.asarray(luz, dtype=float)
        self.fundo_baixo, self.fundo_alto = np.array(fundo_baixo), np.array(fundo_alto)
        self.casas = ([0.30, 0.33, 0.40], [0.19, 0.21, 0.27]) if escuro else \
                     ([0.92, 0.92, 0.94], [0.74, 0.77, 0.83])
        self.raios = {"primarios": 0, "sombra": 0, "reflexo": 0}

    def ceu(self, D):
        f = np.clip(D[:, 1:2] * 0.8 + 0.5, 0, 1)
        return self.fundo_baixo + (self.fundo_alto - self.fundo_baixo) * f

    def na_sombra(self, P, N):
        para_luz = self.luz - P
        dist = np.linalg.norm(para_luz, axis=1)
        L = para_luz / dist[:, None]
        t, _ = self.cena.intersecao(P + N * 10 * EPS, L)
        self.raios["sombra"] += len(P)
        return t < dist, L

    def iluminar(self, P, N, V, base, sombreado, L):
        H = L + V
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        dif = np.maximum((N * L).sum(axis=1), 0.0)
        esp = np.maximum((N * H).sum(axis=1), 0.0) ** BRILHO * (dif > 0)
        luz = (~sombreado)[:, None]
        return KA * base + luz * (KD * base * dif[:, None] + KS * esp[:, None])

    def tracar(self, O, D, profundidade=1):
        cena = self.cena
        t_arv, obj = cena.intersecao(O, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_piso = (PLANO_Y - O[:, 1]) / D[:, 1]
        t_piso = np.where((t_piso > EPS) & np.isfinite(t_piso), t_piso, np.inf)
        cor = self.ceu(D)

        # vasos
        sel = np.isfinite(t_arv) & (t_arv <= t_piso)
        if sel.any():
            P = O[sel] + t_arv[sel, None] * D[sel]
            N = cena.normal(P, obj[sel])
            N = np.where(((N * D[sel]).sum(axis=1) > 0)[:, None], -N, N)
            sombra, L = self.na_sombra(P, N)
            cor[sel] = self.iluminar(P, N, -D[sel], cena.cor[obj[sel]], sombra, L)

        # piso quadriculado com sombra e reflexo
        sel = np.isfinite(t_piso) & (t_piso < t_arv)
        if sel.any():
            P = O[sel] + t_piso[sel, None] * D[sel]
            N = np.tile([0.0, 1.0, 0.0], (len(P), 1))
            casa = (np.floor(P[:, 0] / 0.4) + np.floor(P[:, 2] / 0.4)) % 2
            base = np.where(casa[:, None] == 0, self.casas[0], self.casas[1])
            sombra, L = self.na_sombra(P, N)
            local = self.iluminar(P, N, -D[sel], base, sombra, L)
            if profundidade > 0:
                R = D[sel] - 2 * (D[sel] * N).sum(axis=1, keepdims=True) * N
                self.raios["reflexo"] += len(P)
                refletida = self.tracar(P + N * 10 * EPS, R, profundidade - 1)
                local = (1 - REFLEXO_PISO) * local + REFLEXO_PISO * refletida
            # o piso some aos poucos ao longe, misturando-se com o céu
            longe = np.clip((t_piso[sel] - 3.0) / 6.0, 0, 1)[:, None]
            local = (1 - longe) * local + longe * self.ceu(D[sel])
            cor[sel] = local
        return np.clip(cor, 0, 1)


# ------------------------------------------------------------------ montagem
def montar(args):
    """Cena, câmera e luz a partir da vista passada pelo visualizador."""
    import visualizador as vis                   # só usa matrizes e o Estado (sem OpenGL)
    arvore = ler_vtk(args.modelo)
    estado = vis.Estado(arvore)
    vista = json.loads(args.vista) if args.vista else {
        "modo": 1, "vista": 3, "dist": 3.4, "ang_x": 0.0, "ang_y": 0.0}
    for campo, valor in vista.items():
        setattr(estado, campo, np.array(valor) if isinstance(valor, list) else valor)
    if args.tema:
        vis.trocar_tema(args.tema)

    M = estado.matriz_modelo()
    pontos = (np.hstack([arvore.pontos, np.ones((arvore.n_pontos, 1))]) @ M.T)[:, :3]
    escala_mundo = np.linalg.norm(M[:3, 0])           # a escala uniforme de M
    raios = arvore.raios * vis.MM_PARA_M * estado.exagero * escala_mundo
    valores, vmin, vmax, mapa, _, _ = estado.escala_cor()
    cores = mapear_cores(valores, vmin, vmax, mapa)
    if estado.modo == 2:
        cores = np.tile(vis.VERMELHO, (arvore.n_segmentos, 1))
    cena = CenaArvore(pontos, arvore.segmentos, raios, cores)
    tema = vis.tema
    tracador = Tracador(cena, estado.luz, tema["fundo_baixo"], tema["fundo_alto"],
                        escuro=tema is vis.TEMAS["escuro"])
    PV = estado.matriz_projecao() @ estado.matriz_visao()
    return arvore, cena, tracador, PV


def raios_da_camera(PV, largura, altura, linhas):
    """Raios dos pixels das linhas dadas: inversa de P . V, como no Ray Casting."""
    j, i = np.meshgrid(linhas, np.arange(largura), indexing="ij")
    x = 2.0 * (i + 0.5) / largura - 1.0
    y = 1.0 - 2.0 * (j + 0.5) / altura
    inv = np.linalg.inv(PV)
    def ponto(z):
        h = np.stack([x.ravel(), y.ravel(), np.full(x.size, z), np.ones(x.size)], axis=1) @ inv.T
        return h[:, :3] / h[:, 3:4]
    perto, longe = ponto(-1.0), ponto(1.0)
    D = longe - perto
    return perto, D / np.linalg.norm(D, axis=1, keepdims=True)


def renderizar(args, ao_progresso=None, faixa=24):
    """Imagem (altura, largura, 3) uint8, calculada em faixas horizontais.

    Com --ssaa k, cada pixel recebe k x k raios e a cor final é a média
    (supersampling, o mesmo antialiasing do TP1), ao custo de k^2 vezes o tempo."""
    arvore, cena, tracador, PV = montar(args)
    k = max(1, getattr(args, "ssaa", 1))
    largura = args.largura * k
    altura = largura                              # a janela do visualizador é quadrada
    faixa *= k
    imagem = np.zeros((altura, largura, 3))
    inicio = time.perf_counter()
    for y0 in range(0, altura, faixa):
        linhas = np.arange(y0, min(y0 + faixa, altura))
        O, D = raios_da_camera(PV, largura, altura, linhas)
        tracador.raios["primarios"] += len(O)
        imagem[linhas] = tracador.tracar(O, D, 0 if args.rapido else 1).reshape(len(linhas), largura, 3)
        if ao_progresso and not ao_progresso(reduzir(imagem, k), linhas[-1] + 1, altura):
            break
    segundos = time.perf_counter() - inicio
    return (reduzir(imagem, k) * 255).astype(np.uint8), tracador.raios, segundos, arvore, cena


def reduzir(imagem, k):
    """Média de cada bloco k x k (filtro caixa)."""
    if k == 1:
        return imagem
    h, w = imagem.shape[0] // k, imagem.shape[1] // k
    return imagem.reshape(h, k, w, k, 3).mean(axis=(1, 3))


def main():
    parser = argparse.ArgumentParser(description="Ray Tracer da árvore CCO")
    parser.add_argument("modelo", nargs="?", default="data/tree3D_Nterm128.vtk")
    parser.add_argument("--largura", type=int, default=600)
    parser.add_argument("--rapido", action="store_true", help="sem reflexo no piso")
    parser.add_argument("--ssaa", type=int, default=1, help="raios por pixel = ssaa^2")
    parser.add_argument("--vista", default=None, help="JSON com a vista do visualizador")
    parser.add_argument("--tema", default=None)
    args = parser.parse_args()

    import pygame
    pygame.display.init()
    pygame.font.init()
    L = args.largura
    tela = pygame.display.set_mode((L, L + 30))
    pygame.display.set_caption("Ray Tracer da árvore CCO")
    fonte = pygame.font.Font(None, 22)

    def mostrar(img, legenda):
        tela.fill((20, 35, 60))
        tela.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        tela.blit(fonte.render(legenda, True, (230, 235, 245)), (8, L + 7))
        pygame.display.flip()

    def progresso(imagem, feitas, total):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN
                                              and evento.key == pygame.K_ESCAPE):
                return False
        mostrar((imagem * 255).astype(np.uint8), f"calculando... {100 * feitas // total}%")
        return True

    img, raios, segundos, arvore, cena = renderizar(args, progresso)
    legenda = (f"{arvore.nome}: {cena.m} vasos | {raios['primarios']} primários, "
               f"{raios['sombra']} sombra, {raios['reflexo']} reflexo | {segundos:.1f} s | S salvar")
    print(legenda)
    rodando = True
    while rodando:
        mostrar(img, legenda)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN
                                              and evento.key == pygame.K_ESCAPE):
                rodando = False
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_s:
                pasta = Path("report/figures")
                pasta.mkdir(parents=True, exist_ok=True)
                caminho = pasta / f"rt_{arvore.nome}.png"
                pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), str(caminho))
                print("salvo:", caminho)
        pygame.time.wait(30)
    pygame.quit()


if __name__ == "__main__":
    main()
