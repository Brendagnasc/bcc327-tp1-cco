"""
visualizador.py

Visualizador principal da árvore CCO (pygame-ce + PyOpenGL).
Versão 1: desenha os segmentos como linhas coloridas pelo raio,
com projeção ortográfica ajustada à caixa envolvente do modelo.

Uso:
    python src/visualizador.py [arquivo.vtk]

Teclas:
    C    alterna o mapa de cores (jet / viridis)
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
from leitor_vtk import ler_vtk

LARGURA, ALTURA = 900, 900
FUNDO = (1.0, 1.0, 1.0)


def montar_vertices(arvore, nome_mapa):
    """Gera os arrays que o OpenGL desenha com GL_LINES.

    Cada segmento i -> j vira 2 vértices seguidos, e os dois recebem
    a cor do raio daquele segmento (o raio é atributo de célula).
    """
    vertices = arvore.pontos[arvore.segmentos].reshape(-1, 3)      # (2M, 3)
    cores_seg = mapear_cores(arvore.raios, arvore.raios.min(),
                             arvore.raios.max(), nome_mapa)          # (M, 3)
    cores = np.repeat(cores_seg, 2, axis=0)                          # (2M, 3)
    return vertices.astype(np.float32), cores.astype(np.float32)


def configurar_projecao(arvore):
    """Projeção ortográfica que enquadra a árvore com 10% de margem."""
    minimo = arvore.pontos[:, :2].min(axis=0)
    maximo = arvore.pontos[:, :2].max(axis=0)
    centro = (minimo + maximo) / 2
    meio = (maximo - minimo).max() / 2 * 1.1

    aspecto = LARGURA / ALTURA
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(centro[0] - meio * aspecto, centro[0] + meio * aspecto,
            centro[1] - meio, centro[1] + meio, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def desenhar(vertices, cores):
    glClear(GL_COLOR_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, cores)
    glDrawArrays(GL_LINES, 0, len(vertices))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree2D_Nterm064.vtk"
    arvore = ler_vtk(caminho)

    pygame.init()
    pygame.display.set_mode((LARGURA, ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)

    mapas = ["jet", "viridis"]
    mapa_atual = 0
    vertices, cores = montar_vertices(arvore, mapas[mapa_atual])
    pygame.display.set_caption(f"TP1 CCO - {arvore.nome} - mapa {mapas[mapa_atual]}")

    glClearColor(*FUNDO, 1.0)
    glLineWidth(2.0)
    configurar_projecao(arvore)

    relogio = pygame.time.Clock()
    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key == pygame.K_c:
                    mapa_atual = (mapa_atual + 1) % len(mapas)
                    vertices, cores = montar_vertices(arvore, mapas[mapa_atual])
                    pygame.display.set_caption(
                        f"TP1 CCO - {arvore.nome} - mapa {mapas[mapa_atual]}")

        desenhar(vertices, cores)
        pygame.display.flip()
        relogio.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
