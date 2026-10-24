"""
visualizador.py

Visualizador principal da árvore CCO (pygame-ce + PyOpenGL).
Versão 2: três modos de representação e barra de cores.

Uso:
    python src/visualizador.py [arquivo.vtk]

Teclas:
    E    alterna o modo: linhas coloridas / espessura + cor / espessura cor sólida
    C    alterna o mapa de cores (jet / viridis)
    + -  aumenta / diminui o fator de exagero da espessura
    ESC  sai
"""

import os
import sys

# No WSL (WSLg) a variável WAYLAND_DISPLAY existe e o PyOpenGL tentaria usar EGL,
# mas o pygame cria o contexto via X11/GLX. Forçar GLX evita esse conflito.
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pygame
from OpenGL.GL import *

from cores import mapear_cores
from geometria import faixas_2d
from leitor_vtk import ler_vtk

LARGURA, ALTURA = 900, 900
FUNDO = (1.0, 1.0, 1.0)
VERMELHO = (0.8, 0.1, 0.1)
MODOS = ["linhas + cor", "espessura + cor", "espessura sólida"]
MAPAS = ["jet", "viridis"]
MM_PARA_M = 1e-3   # raio vem em mm; coordenadas em m


class Estado:
    def __init__(self, arvore):
        self.arvore = arvore
        self.modo = 0
        self.mapa = 0
        self.exagero = 2.0     # espessura desenhada = raio real x exagero
        self.vertices = self.cores = None
        self.primitiva = GL_LINES

    def reconstruir(self):
        """Recalcula vértices e cores sempre que o modo, mapa ou exagero muda."""
        arv = self.arvore
        cores_seg = mapear_cores(arv.raios, arv.raios.min(), arv.raios.max(),
                                 MAPAS[self.mapa])
        if self.modo == 2:
            cores_seg = np.tile(VERMELHO, (arv.n_segmentos, 1))

        if self.modo == 0:
            self.vertices = arv.pontos[arv.segmentos].reshape(-1, 3)
            self.cores = np.repeat(cores_seg, 2, axis=0)
            self.primitiva = GL_LINES
        else:
            meia = arv.raios * MM_PARA_M * self.exagero
            self.vertices, seg_id = faixas_2d(arv.pontos, arv.segmentos, meia)
            self.cores = cores_seg[seg_id]
            self.primitiva = GL_TRIANGLES

        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.cores = np.ascontiguousarray(self.cores, dtype=np.float32)
        pygame.display.set_caption(
            f"TP1 CCO - {arv.nome} - {MODOS[self.modo]} - mapa {MAPAS[self.mapa]}"
            f" - exagero {self.exagero:.1f}x")


def configurar_projecao(arvore):
    """Projeção ortográfica que enquadra a árvore com margem.

    O centro da janela sobe um pouco para a árvore descer na tela
    e deixar espaço livre para a barra de cores no topo.
    """
    minimo = arvore.pontos[:, :2].min(axis=0)
    maximo = arvore.pontos[:, :2].max(axis=0)
    centro = (minimo + maximo) / 2
    meio = (maximo - minimo).max() / 2 * 1.2
    centro[1] += meio * 0.1

    aspecto = LARGURA / ALTURA
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(centro[0] - meio * aspecto, centro[0] + meio * aspecto,
            centro[1] - meio, centro[1] + meio, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def desenhar_arvore(estado):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, estado.vertices)
    glColorPointer(3, GL_FLOAT, 0, estado.cores)
    glDrawArrays(estado.primitiva, 0, len(estado.vertices))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)


def desenhar_texto(fonte, texto, x, y):
    """Escreve texto na tela em coordenadas de pixel (origem embaixo à esquerda)."""
    superficie = fonte.render(texto, True, (20, 20, 20))
    dados = pygame.image.tobytes(superficie, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(superficie.get_width(), superficie.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, dados)
    return superficie.get_width()


def desenhar_barra_cores(estado, fonte):
    """Barra horizontal no topo da tela, desenhada em coordenadas de pixel."""
    arv = estado.arvore
    x0, x1, y0, y1 = LARGURA * 0.3, LARGURA * 0.7, ALTURA - 60, ALTURA - 40
    passos = 64
    valores = np.linspace(arv.raios.min(), arv.raios.max(), passos + 1)
    cores = mapear_cores(valores, valores[0], valores[-1], MAPAS[estado.mapa])
    xs = np.linspace(x0, x1, passos + 1)

    # Troca temporariamente para uma projeção em pixels
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, LARGURA, 0, ALTURA, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    glBegin(GL_QUAD_STRIP)
    for x, cor in zip(xs, cores):
        glColor3f(*cor)
        glVertex2f(x, y0)
        glVertex2f(x, y1)
    glEnd()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    desenhar_texto(fonte, f"{arv.raios.min():.3f}", x0 - 55, y0 + 2)
    desenhar_texto(fonte, f"{arv.raios.max():.3f}", x1 + 8, y0 + 2)
    desenhar_texto(fonte, "raio (mm)", LARGURA / 2 - 35, y1 + 6)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree2D_Nterm064.vtk"
    arvore = ler_vtk(caminho)

    pygame.init()
    pygame.display.set_mode((LARGURA, ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    fonte = pygame.font.Font(None, 24)

    glClearColor(*FUNDO, 1.0)
    glLineWidth(2.0)
    glEnable(GL_BLEND)                                  # para o texto com transparência
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    configurar_projecao(arvore)

    estado = Estado(arvore)
    estado.reconstruir()

    relogio = pygame.time.Clock()
    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key == pygame.K_e:
                    estado.modo = (estado.modo + 1) % len(MODOS)
                    estado.reconstruir()
                elif evento.key == pygame.K_c:
                    estado.mapa = (estado.mapa + 1) % len(MAPAS)
                    estado.reconstruir()
                elif evento.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    estado.exagero = min(estado.exagero + 0.5, 10.0)
                    estado.reconstruir()
                elif evento.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    estado.exagero = max(estado.exagero - 0.5, 0.5)
                    estado.reconstruir()

        glClear(GL_COLOR_BUFFER_BIT)
        desenhar_arvore(estado)
        if estado.modo != 2:
            desenhar_barra_cores(estado, fonte)
        pygame.display.flip()
        relogio.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
