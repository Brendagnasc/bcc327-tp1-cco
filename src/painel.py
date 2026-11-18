"""
painel.py

Painel de controles com botões, seletores e sliders, desenhado pelo grupo
com OpenGL (retângulos) e pygame (texto). Substitui o teclado para quem não
conhece os atalhos.

Funciona no estilo "modo imediato" (immediate mode GUI): a cada quadro o
painel é redesenhado do zero, e cada controle é uma chamada de função que
desenha e, ao mesmo tempo, registra a sua área clicável e o que fazer
quando for clicado. No quadro seguinte, um clique do mouse é comparado com
essas áreas. Não há objetos de "botão" guardados entre quadros.

    painel.inicio()
    painel.secao("Animação")
    painel.botoes([("Crescer", False, acao), ("Fluxo", estado.fluxo, acao2)])
    painel.slider("Espessura", valor, 0.5, 10, "{:.1f}x", ao_mudar)
    painel.fim()
"""

import numpy as np
import pygame
from OpenGL.GL import *

LARGURA_PAINEL = 300
LINHA = 25                 # altura de cada linha de controles (pixels)
ROTULO = 86                # largura da coluna de rótulos
MARGEM = 12

CORES = {
    True: {   # tema escuro
        "fundo": (0.08, 0.10, 0.15, 0.90), "controle": (0.20, 0.24, 0.32),
        "sobre": (0.28, 0.33, 0.44), "ativo": (1.0, 0.78, 0.25),
        "texto": (230, 234, 242), "texto_ativo": (25, 25, 25), "secao": (255, 205, 90),
        "trilho": (0.30, 0.35, 0.45),
    },
    False: {  # tema claro
        "fundo": (0.96, 0.97, 0.99, 0.94), "controle": (0.84, 0.87, 0.92),
        "sobre": (0.76, 0.80, 0.88), "ativo": (0.95, 0.68, 0.10),
        "texto": (25, 30, 40), "texto_ativo": (20, 20, 20), "secao": (150, 90, 0),
        "trilho": (0.70, 0.74, 0.82),
    },
}


class Painel:
    def __init__(self, largura_tela, altura_tela, desenhar_texto):
        self.W, self.H = largura_tela, altura_tela
        self.texto = desenhar_texto
        self.fonte = pygame.font.Font(None, 21)
        self.fonte_secao = pygame.font.Font(None, 22)
        self.fonte_secao.set_bold(True)
        self.aberto = False
        self.areas = []            # (retângulo, ação) registrados no quadro atual
        self.areas_anteriores = []
        self.slider_ativo = None   # (retângulo, mín, máx, ao_mudar, ao_soltar)
        self.mouse = (-1, -1)
        self.altura_conteudo = 700     # medida no quadro anterior, para o fundo do painel

    # -------------------------------------------------------------- geometria
    def _gl_retangulo(self, x, y, w, h, cor):
        """Retângulo em coordenadas do pygame (origem no canto superior esquerdo)."""
        y_gl = self.H - y - h
        if len(cor) == 4:
            glColor4f(*cor)
        else:
            glColor3f(*cor)
        glBegin(GL_QUADS)
        glVertex2f(x, y_gl); glVertex2f(x + w, y_gl)
        glVertex2f(x + w, y_gl + h); glVertex2f(x, y_gl + h)
        glEnd()

    def _escrever(self, texto, x, y, cor, fonte=None, centro_w=None):
        fonte = fonte or self.fonte
        largura, altura = fonte.size(texto)
        if centro_w is not None:
            x += (centro_w - largura) / 2
        self.texto(fonte, texto, x, self.H - y - altura, cor=cor)

    def _sob_mouse(self, x, y, w, h):
        mx, my = self.mouse
        return x <= mx < x + w and y <= my < y + h

    # -------------------------------------------------------------- quadro
    def inicio(self, escuro):
        """Prepara o quadro: projeção em pixels e o botão Menu (sempre visível)."""
        self.c = CORES[escuro]
        self.mouse = pygame.mouse.get_pos()
        self.areas = []
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.W, 0, self.H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # botão que abre e fecha o painel
        w, h = 84, 28
        cor = self.c["ativo"] if self.aberto else (
            self.c["sobre"] if self._sob_mouse(8, 8, w, h) else self.c["controle"])
        self._gl_retangulo(8, 8, w, h, cor)
        self._escrever("Fechar" if self.aberto else "Menu", 8, 13,
                       self.c["texto_ativo"] if self.aberto else self.c["texto"], centro_w=w)
        self.areas.append(((8, 8, w, h), self.alternar))

        if self.aberto:
            self.topo = 44
            self.base = min(self.topo + self.altura_conteudo, self.H - 130)
            self._gl_retangulo(0, self.topo, LARGURA_PAINEL, self.base - self.topo, self.c["fundo"])
            self.y = self.topo + 8
        return self.aberto

    def fim(self):
        self.areas_anteriores = self.areas
        if self.aberto:
            self.altura_conteudo = self.y - self.topo + 8

    def alternar(self):
        self.aberto = not self.aberto

    # -------------------------------------------------------------- controles
    def secao(self, titulo):
        self.y += 4
        self._escrever(titulo, MARGEM, self.y + 3, self.c["secao"], self.fonte_secao)
        self.y += LINHA - 3

    def _celulas(self, itens, x0):
        """Divide a largura disponível em células iguais e desenha cada uma."""
        largura = LARGURA_PAINEL - MARGEM - x0
        w = largura / len(itens)
        for i, (rotulo, ativo, acao) in enumerate(itens):
            x = x0 + i * w
            caixa = (x + 1, self.y, w - 2, LINHA - 4)
            cor = self.c["ativo"] if ativo else (
                self.c["sobre"] if self._sob_mouse(*caixa) else self.c["controle"])
            self._gl_retangulo(*caixa, cor)
            self._escrever(rotulo, x + 1, self.y + 4,
                           self.c["texto_ativo"] if ativo else self.c["texto"], centro_w=w - 2)
            self.areas.append((caixa, acao))
        self.y += LINHA

    def botoes(self, itens):
        """Linha de botões. itens: [(rótulo, aceso, ação)]; aceso marca um liga/desliga."""
        self._celulas(itens, MARGEM)

    def segmentos(self, rotulo, opcoes, atual, ao_escolher):
        """Escolha exclusiva entre opções (só uma acesa)."""
        self._escrever(rotulo, MARGEM, self.y + 4, self.c["texto"])
        itens = [(nome, i == atual, (lambda i=i: ao_escolher(i))) for i, nome in enumerate(opcoes)]
        self._celulas(itens, MARGEM + ROTULO)

    def ciclo(self, rotulo, texto, anterior, proximo):
        """Seletor com setas: < texto >."""
        self._escrever(rotulo, MARGEM, self.y + 4, self.c["texto"])
        x0, x1 = MARGEM + ROTULO, LARGURA_PAINEL - MARGEM
        h = LINHA - 4
        for x, simbolo, acao in ((x0, "<", anterior), (x1 - 26, ">", proximo)):
            caixa = (x, self.y, 26, h)
            self._gl_retangulo(*caixa, self.c["sobre"] if self._sob_mouse(*caixa) else self.c["controle"])
            self._escrever(simbolo, x, self.y + 4, self.c["texto"], centro_w=26)
            self.areas.append((caixa, acao))
        self._escrever(texto, x0 + 26, self.y + 4, self.c["texto"], centro_w=x1 - x0 - 52)
        self.y += LINHA

    def slider(self, rotulo, valor, vmin, vmax, formato, ao_mudar, ao_soltar=None):
        """Trilho com um botão arrastável; ao_mudar recebe o novo valor."""
        self._escrever(rotulo, MARGEM, self.y + 4, self.c["texto"])
        texto_valor = formato.format(valor).replace(".", ",")
        largura_valor = 54
        x0 = MARGEM + ROTULO
        largura = LARGURA_PAINEL - MARGEM - x0 - largura_valor
        trilho = (x0, self.y + LINHA / 2 - 5, largura, 6)
        self._gl_retangulo(*trilho, self.c["trilho"])
        f = 0.0 if vmax == vmin else float(np.clip((valor - vmin) / (vmax - vmin), 0, 1))
        self._gl_retangulo(x0, trilho[1], largura * f, 6, self.c["ativo"])
        self._gl_retangulo(x0 + largura * f - 6, self.y + 2, 12, LINHA - 8, self.c["ativo"])
        self._escrever(texto_valor, x0 + largura + 6, self.y + 4, self.c["texto"])

        area = (x0 - 6, self.y, largura + 12, LINHA - 2)
        dados = ((x0, largura), vmin, vmax, ao_mudar, ao_soltar)
        self.areas.append((area, ("slider", dados)))
        self.y += LINHA

    # -------------------------------------------------------------- mouse
    def clique(self, pos):
        """True se o clique caiu no painel (e então não deve girar a árvore)."""
        x, y = pos
        for (ax, ay, aw, ah), acao in self.areas_anteriores:
            if ax <= x < ax + aw and ay <= y < ay + ah:
                if isinstance(acao, tuple) and acao[0] == "slider":
                    self.slider_ativo = acao[1]
                    self.mover(pos)
                else:
                    acao()
                return True
        return self.aberto and x < LARGURA_PAINEL and 44 <= y < 44 + self.altura_conteudo

    def mover(self, pos):
        if self.slider_ativo is None:
            return False
        (x0, largura), vmin, vmax, ao_mudar, _ = self.slider_ativo
        f = float(np.clip((pos[0] - x0) / largura, 0.0, 1.0))
        ao_mudar(vmin + f * (vmax - vmin))
        return True

    def soltar(self):
        if self.slider_ativo is None:
            return False
        ao_soltar = self.slider_ativo[4]
        self.slider_ativo = None
        if ao_soltar:
            ao_soltar()
        return True
