"""
modo_didatico.py

Modo didático: uma grade de pixels "gigantes" (um framebuffer em NumPy)
onde só os algoritmos do grupo escrevem. Serve para mostrar, pixel a pixel,
o resultado da rasterização (e, nos próximos passos, do preenchimento,
do antialiasing e do recorte).

Usa apenas pygame-ce 2D: nenhum algoritmo de desenho da biblioteca é usado
para gerar os pixels; o pygame só amplia o framebuffer para a tela.

Uso:
    python src/modo_didatico.py [arquivo_2D.vtk]

Teclas e mouse:
    1        modo árvore: segmentos da árvore CCO rasterizados
    2        modo segmento: clique em dois pixels para traçar um segmento
    B        alterna o algoritmo: Bresenham / DDA
    L        mostra / esconde a reta ideal (contínua) por cima dos pixels
    S        salva a tela em report/figures/
    ESC      sai
"""

import sys
from pathlib import Path

import numpy as np
import pygame

from cores import mapear_cores
from leitor_vtk import ler_vtk
from rasterizacao import bresenham, dda, imprimir_rastro

COLUNAS, LINHAS = 80, 60          # tamanho do framebuffer, em pixels
CELULA = 12                       # cada pixel vira um quadrado de 12x12 na tela
PAINEL = 60                       # faixa inferior para texto
LARGURA, ALTURA = COLUNAS * CELULA, LINHAS * CELULA + PAINEL
BRANCO = (255, 255, 255)
ALGORITMOS = {"Bresenham": bresenham, "DDA": dda}


class Framebuffer:
    """Matriz LINHAS x COLUNAS x RGB. Origem (0, 0) no canto inferior esquerdo,
    y para cima, como nos livros de Computação Gráfica."""

    def __init__(self):
        self.pixels = np.full((LINHAS, COLUNAS, 3), 255, dtype=np.uint8)

    def limpar(self):
        self.pixels[:] = 255

    def acender(self, lista, cor):
        for x, y in lista:
            if 0 <= x < COLUNAS and 0 <= y < LINHAS:
                self.pixels[y, x] = cor

    def para_superficie(self):
        # inverte as linhas (y para cima -> y para baixo da tela) e amplia
        imagem = self.pixels[::-1].swapaxes(0, 1)
        pequena = pygame.surfarray.make_surface(imagem)
        return pygame.transform.scale(pequena, (COLUNAS * CELULA, LINHAS * CELULA))


def centro_na_tela(x, y):
    """Centro do pixel (x, y) da grade em coordenadas da janela."""
    return ((x + 0.5) * CELULA, (LINHAS - 1 - y + 0.5) * CELULA)


def pixel_do_mouse(pos):
    mx, my = pos
    return int(mx // CELULA), LINHAS - 1 - int(my // CELULA)


def segmentos_da_arvore(arvore, margem=3):
    """Projeta a árvore 2D na grade: escala uniforme + arredondamento."""
    xy = arvore.pontos[:, :2]
    minimo = xy.min(axis=0)
    tamanho = (xy.max(axis=0) - minimo).max()
    util = min(COLUNAS, LINHAS) - 1 - 2 * margem
    grade = np.rint((xy - minimo) / tamanho * util).astype(int) + margem
    grade[:, 0] += (COLUNAS - LINHAS) // 2          # centraliza na horizontal
    return [(*grade[i], *grade[j]) for i, j in arvore.segmentos]


class App:
    def __init__(self, arvore):
        self.fb = Framebuffer()
        self.arvore = arvore
        self.segs_arvore = segmentos_da_arvore(arvore)
        self.cores_arvore = (mapear_cores(arvore.raios, arvore.raios.min(),
                                          arvore.raios.max(), "jet") * 255).astype(np.uint8)
        self.modo = 1
        self.algoritmo = "Bresenham"
        self.reta_ideal = True
        self.segmento = (20, 10, 30, 18)            # exemplo do relatório
        self.clique = None
        self.info = ""

    # ------------------------------------------------------------ rasterizar
    def rasterizar(self):
        self.fb.limpar()
        f = ALGORITMOS[self.algoritmo]
        total = 0
        if self.modo == 1:
            # finos primeiro, grossos por cima
            for idx in np.argsort(self.arvore.raios):
                px = f(*self.segs_arvore[idx])
                self.fb.acender(px, self.cores_arvore[idx])
                total += len(px)
            self.info = (f"Árvore {self.arvore.nome} | {self.algoritmo} | "
                         f"{len(self.segs_arvore)} segmentos | {total} pixels")
        else:
            x0, y0, x1, y1 = self.segmento
            px = f(x0, y0, x1, y1)
            self.fb.acender(px, (30, 60, 160))
            self.fb.acender([(x0, y0)], (0, 160, 0))
            self.fb.acender([(x1, y1)], (200, 0, 0))
            self.info = (f"Segmento ({x0},{y0}) a ({x1},{y1}) | {self.algoritmo} | "
                         f"{len(px)} pixels | clique em dois pixels")

    # ------------------------------------------------------------ desenhar
    def desenhar(self, tela, fonte):
        tela.fill(BRANCO)
        tela.blit(self.fb.para_superficie(), (0, 0))

        # linhas da grade
        cinza = (215, 215, 215)
        for c in range(COLUNAS + 1):
            pygame.draw.line(tela, cinza, (c * CELULA, 0), (c * CELULA, LINHAS * CELULA))
        for l in range(LINHAS + 1):
            pygame.draw.line(tela, cinza, (0, l * CELULA), (COLUNAS * CELULA, l * CELULA))

        # reta ideal por cima, só como referência visual
        if self.reta_ideal:
            segs = self.segs_arvore if self.modo == 1 else [self.segmento]
            for x0, y0, x1, y1 in segs:
                pygame.draw.aaline(tela, (0, 0, 0), centro_na_tela(x0, y0),
                                   centro_na_tela(x1, y1))

        if self.clique is not None:
            pygame.draw.rect(tela, (0, 160, 0), (*[v - CELULA / 2 for v in
                             centro_na_tela(*self.clique)], CELULA, CELULA), 2)

        y_txt = LINHAS * CELULA + 8
        tela.blit(fonte.render(self.info, True, (20, 20, 20)), (10, y_txt))
        ajuda = "1 árvore  2 segmento  B algoritmo  L reta ideal  S salvar  ESC sair"
        tela.blit(fonte.render(ajuda, True, (90, 90, 90)), (10, y_txt + 24))

    def salvar(self, tela):
        pasta = Path("report/figures")
        pasta.mkdir(parents=True, exist_ok=True)
        nome = "raster_arvore" if self.modo == 1 else "raster_segmento"
        caminho = pasta / f"{nome}_{self.algoritmo.lower()}.png"
        pygame.image.save(tela, str(caminho))
        print("salvo:", caminho)

    # ------------------------------------------------------------ eventos
    def tratar_clique(self, pos):
        if self.modo != 2 or pos[1] >= LINHAS * CELULA:
            return
        p = pixel_do_mouse(pos)
        if self.clique is None:
            self.clique = p
        else:
            self.segmento = (*self.clique, *p)
            self.clique = None
            self.rasterizar()
            if self.algoritmo == "Bresenham":
                imprimir_rastro(*self.segmento)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree2D_Nterm016.vtk"
    app = App(ler_vtk(caminho))

    pygame.display.init()             # só vídeo e fonte (evita iniciar o áudio)
    pygame.font.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("TP1 CCO - modo didático")
    fonte = pygame.font.Font(None, 24)
    app.rasterizar()

    relogio = pygame.time.Clock()
    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                app.tratar_clique(evento.pos)
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key in (pygame.K_1, pygame.K_2):
                    app.modo = evento.key - pygame.K_0
                    app.clique = None
                    app.rasterizar()
                elif evento.key == pygame.K_b:
                    app.algoritmo = "DDA" if app.algoritmo == "Bresenham" else "Bresenham"
                    app.rasterizar()
                elif evento.key == pygame.K_l:
                    app.reta_ideal = not app.reta_ideal
                elif evento.key == pygame.K_s:
                    app.salvar(tela)

        app.desenhar(tela, fonte)
        pygame.display.flip()
        relogio.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
