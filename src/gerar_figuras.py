"""
gerar_figuras.py

Gera, de uma vez, todas as figuras do relatório e da apresentação do TP2,
com nomes fixos em report/figures/, garantindo que sejam reproduzíveis.

Uso (a partir da raiz do repositório):
    python src/gerar_figuras.py

Uma janela OpenGL abre por alguns segundos enquanto as capturas são feitas.
"""

import contextlib
import io
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pygame
from OpenGL.GL import *

import raytracer
import visualizador as vis
from leitor_vtk import ler_vtk

PASTA = Path("report/figures")
fonte = None


def novo_estado(arquivo, **ajustes):
    """Estado do visualizador com os campos ajustados e a geometria pronta."""
    estado = vis.Estado(ler_vtk(f"data/{arquivo}.vtk"))
    estado.fbo_msaa, estado.amostras_msaa = vis.criar_fbo_msaa(4)
    estado.modo = 1 if estado.arvore.eh_3d else 0
    estado.eixos = False
    estado.painel = False                    # painel de métricas só nas figuras do Módulo 8
    for chave, valor in ajustes.items():
        setattr(estado, chave, valor)
    estado.reconstruir()
    return estado


def salvar(estado, nome):
    vis.desenhar_quadro(estado, fonte)
    vis.capturar(PASTA / f"{nome}.png")
    pygame.display.flip()


def figura(nome, arquivo="tree3D_Nterm128", **ajustes):
    salvar(novo_estado(arquivo, **ajustes), nome)


def pixel_do_vaso(estado, k):
    """Pixel onde aparece o ponto médio do vaso k (para simular o clique)."""
    a, b = estado.arvore.pontos[estado.arvore.segmentos[k]]
    c = estado.matriz_projecao() @ estado.matriz_visao() @ estado.matriz_modelo() \
        @ np.append((a + b) / 2, 1.0)
    c = c / c[3]
    return (int(round((c[0] + 1) / 2 * vis.LARGURA - 0.5)),
            int(round((1 - c[1]) / 2 * vis.ALTURA - 0.5)))


def figuras_visualizador():
    padrao = dict(vista=3, dist=2.6)

    # Módulo 1: geometria 3D e vistas
    figura("m1_tubos", "tree3D_Nterm256", vista=3, dist=3.4)
    for i, vista in enumerate(["frontal", "lateral", "superior"]):
        figura(f"m1_vista_{vista}", "tree3D_Nterm256", vista=i, dist=3.4, plano=False)

    # Módulo 2: visibilidade
    figura("m2_zbuffer_sem_culling", "tree3D_Nterm064", vista=3, dist=2.0, plano=False,
           destacar_tras=True)
    figura("m2_sem_zbuffer_sem_culling", "tree3D_Nterm064", vista=3, dist=2.0, plano=False,
           destacar_tras=True, zbuffer=False)
    figura("m2_sem_zbuffer_com_culling", "tree3D_Nterm064", vista=3, dist=2.0, plano=False,
           destacar_tras=True, zbuffer=False, culling=1)
    figura("m2_zbuffer_culling_proprio", "tree3D_Nterm064", vista=3, dist=2.0, plano=False,
           destacar_tras=True, culling=2)
    figura("m2_profundidade", "tree3D_Nterm064", vista=3, dist=2.0, plano=False,
           ver_profundidade=True)

    # Módulo 3: sombreamento, componentes e luz (de perto, para ver as facetas)
    perto = dict(vista=3, dist=1.2, plano=False, msaa=True)
    for nome in ("Flat", "Gouraud", "Phong"):
        figura(f"m3_{nome.lower()}", "tree3D_Nterm064", sombreamento=nome, **perto)
    for c, nome in enumerate(["ambiente", "difusa", "especular"]):
        figura(f"m3_{nome}", "tree3D_Nterm064", componentes=c, **perto)
    figura("m3_luz_esquerda", **padrao, luz=np.array([-1.8, 1.2, 1.5]))
    figura("m3_luz_direita", **padrao, luz=np.array([1.8, -0.6, 1.5]))

    # Módulo 4: transparência
    figura("m4_opaco", **padrao, eixos=True)
    figura("m4_alfa06_painter", **padrao, eixos=True, opacidade=1)
    figura("m4_alfa03_painter", **padrao, eixos=True, opacidade=2)
    figura("m4_alfa03_sem_ordem", **padrao, eixos=True, opacidade=2, ordem=1)
    figura("m4_alfa03_painter_culling", **padrao, eixos=True, opacidade=2, culling=2)

    # Módulo 5: textura
    figura("m5_plano", vista=3, dist=3.6)
    figura("m5_coordenadas_uv", vista=3, dist=3.6, filtro=3)
    for f, nome in enumerate(["nearest", "bilinear_opengl", "bilinear_propria"]):
        figura(f"m5_{nome}", modo=0, vista=2, dist=1.3, filtro=f)

    # Módulo 6: Ray Casting (clique simulado no ponto médio de um vaso grosso)
    estado = novo_estado("tree3D_Nterm128", **padrao, plano=False)
    for k in np.argsort(estado.arvore.raios)[::-1][3:60]:
        with contextlib.redirect_stdout(io.StringIO()):
            estado.selecionar(*pixel_do_vaso(estado, k))
        if estado.selecionado == k:
            break
    salvar(estado, "m6_selecao")

    # Módulo 8: filtro por raio e métricas
    raios = ler_vtk("data/tree3D_Nterm256.vtk").raios
    figura("m8_completo", "tree3D_Nterm256", vista=3, dist=3.2, painel=True)
    figura("m8_grossos", "tree3D_Nterm256", vista=3, dist=3.2, painel=True,
           filtro_min=float(np.quantile(raios, 0.75)))
    figura("m8_finos", "tree3D_Nterm256", vista=3, dist=3.2, painel=True, culling=2,
           opacidade=1, filtro_max=float(np.quantile(raios, 0.4)))


def figuras_raytracer():
    # Módulo 7: os três modos do mini Ray Tracer
    for modo in raytracer.MODOS:
        img, st = raytracer.renderizar(modo, 640, 480)
        caminho = PASTA / f"m7_rt_modo{modo}.png"
        pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), str(caminho))
        print("salvo:", caminho, st)


if __name__ == "__main__":
    PASTA.mkdir(parents=True, exist_ok=True)
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((vis.LARGURA, vis.ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gerando figuras...")
    fonte = pygame.font.Font(None, 24)
    vis.inicializar_gl()
    figuras_visualizador()
    figuras_raytracer()
    pygame.display.quit()
    print(f"\n{len(list(PASTA.glob('*.png')))} figuras em {PASTA}/")
