"""
gerar_figuras.py

Gera, de uma vez, todas as figuras usadas no relatório e na apresentação,
com nomes fixos em report/figures/. Garante que as figuras são
reproduzíveis a partir do código.

Uso (a partir da raiz do repositório):
    python src/gerar_figuras.py

Uma janela OpenGL abre por alguns segundos enquanto as capturas do
visualizador são feitas; as do modo didático são desenhadas fora da tela.
"""

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import pygame
from OpenGL.GL import *

import modo_didatico as md
import visualizador as vis
from leitor_vtk import ler_vtk

PASTA = Path("report/figures")


# ----------------------------------------------------------------- visualizador
def figura_vis(nome, arquivo, fonte, **ajustes):
    """Cria um Estado do visualizador, aplica os ajustes, desenha e salva."""
    estado = vis.Estado(ler_vtk(f"data/{arquivo}.vtk"))
    estado.fbo_msaa, estado.amostras_msaa = vis.criar_fbo_msaa(4)
    estado.eixos = False
    for chave, valor in ajustes.items():
        setattr(estado, chave, valor)
    estado.reconstruir()
    vis.desenhar_quadro(estado, fonte)
    vis.capturar(PASTA / f"{nome}.png")
    pygame.display.flip()


def recortar_ampliar(origem, destino, caixa, fator=3):
    """Recorta uma região de uma figura e amplia sem suavizar (para ver os pixels)."""
    imagem = pygame.image.load(str(PASTA / origem))
    parte = imagem.subsurface(pygame.Rect(caixa)).copy()
    grande = pygame.transform.scale(parte, (caixa[2] * fator, caixa[3] * fator))
    pygame.image.save(grande, str(PASTA / destino))
    print("salvo:", PASTA / destino)


def figuras_visualizador():
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((vis.LARGURA, vis.ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gerando figuras...")
    fonte = pygame.font.Font(None, 24)
    vis.inicializar_gl()

    # Módulo 1: primitivas e atributos
    figura_vis("m1_linhas_jet", "tree2D_Nterm064", fonte, modo=0)
    figura_vis("m1_faixas_jet", "tree2D_Nterm064", fonte, modo=1)
    figura_vis("m1_faixas_viridis", "tree2D_Nterm064", fonte, modo=1, mapa=1)
    figura_vis("m1_solido", "tree2D_Nterm064", fonte, modo=2)
    figura_vis("m1_tubos_3d", "tree3D_Nterm256", fonte, modo=1, vista=3, dist=3.4)

    # Módulo 2: transformações (eixos do mundo ficam parados)
    figura_vis("m2_original", "tree2D_Nterm064", fonte, modo=1, eixos=True)
    figura_vis("m2_translacao", "tree2D_Nterm064", fonte, modo=1, eixos=True, tx=0.35, ty=-0.2)
    figura_vis("m2_rotacao", "tree2D_Nterm064", fonte, modo=1, eixos=True, ang_z=35)
    figura_vis("m2_escala", "tree2D_Nterm064", fonte, modo=1, eixos=True, s=0.6)

    # Módulo 2: vistas do observador e projeções
    for i, vista in enumerate(["frontal", "lateral", "superior", "isometrica"]):
        figura_vis(f"m2_vista_{vista}", "tree3D_Nterm256", fonte, vista=i, eixos=True)
    figura_vis("m2_perspectiva", "tree3D_Nterm064", fonte, modo=1, vista=3, dist=2.4, eixos=True)
    figura_vis("m2_ortografica", "tree3D_Nterm064", fonte, modo=1, vista=3, dist=2.4, eixos=True,
               perspectiva=False)

    # Módulo 5: MSAA (recorte ampliado de uma borda de tubo)
    figura_vis("m5_msaa_off", "tree3D_Nterm064", fonte, modo=1, vista=3, dist=1.3)
    figura_vis("m5_msaa_on", "tree3D_Nterm064", fonte, modo=1, vista=3, dist=1.3, msaa=True)
    caixa = (380, 330, 150, 150)
    recortar_ampliar("m5_msaa_off.png", "m5_msaa_off_zoom.png", caixa)
    recortar_ampliar("m5_msaa_on.png", "m5_msaa_on_zoom.png", caixa)

    pygame.display.quit()


# ----------------------------------------------------------------- modo didático
def figura_did(app, superficie, fonte, nome, **ajustes):
    for chave, valor in ajustes.items():
        setattr(app, chave, valor)
    app.rasterizar()
    app.desenhar(superficie, fonte)
    pygame.image.save(superficie, str(PASTA / f"{nome}.png"))
    print("salvo:", PASTA / f"{nome}.png")


def figuras_didatico():
    pygame.font.init()
    fonte = pygame.font.Font(None, 22)
    tela = pygame.Surface((md.LARGURA, md.ALTURA))     # desenha fora da tela

    app = md.App(ler_vtk("data/tree2D_Nterm016.vtk"))
    # Módulo 3: rasterização de segmentos
    figura_did(app, tela, fonte, "m3_arvore_bresenham", modo=1)
    figura_did(app, tela, fonte, "m3_arvore_dda", modo=1, algoritmo="DDA")
    figura_did(app, tela, fonte, "m3_segmento_bresenham", modo=2, algoritmo="Bresenham")
    # Módulo 4: Scanline Fill (completo e no meio da animação)
    figura_did(app, tela, fonte, "m4_scanline", modo=3)
    figura_did(app, tela, fonte, "m4_scanline_animacao", modo=3, anim=24)
    app.anim = None
    # Módulo 5: supersampling
    figura_did(app, tela, fonte, "m5_ssaa_leque", modo=4, cena_aa="leque", fator=4)
    figura_did(app, tela, fonte, "m5_ssaa_arvore", modo=4, cena_aa="árvore", fator=4)

    # Módulos 6 e 7 com um modelo um pouco maior
    app = md.App(ler_vtk("data/tree2D_Nterm032.vtk"))
    figura_did(app, tela, fonte, "m6_recorte_antes", modo=5, vista_recorte=1)
    figura_did(app, tela, fonte, "m6_recorte_depois", modo=5, vista_recorte=2)
    figura_did(app, tela, fonte, "m6_recorte_sobreposto", modo=5, vista_recorte=0)

    app = md.App(ler_vtk("data/tree2D_Nterm016.vtk"))
    figura_did(app, tela, fonte, "m7_sh_regiao_retangulo", modo=6, vista_recorte=0)
    figura_did(app, tela, fonte, "m7_sh_regiao_hexagono", tipo_janela="hexágono")
    figura_did(app, tela, fonte, "m7_u_antes", sujeito="forma em U", tipo_janela="retângulo",
               vista_recorte=1)
    figura_did(app, tela, fonte, "m7_sh_u_depois", vista_recorte=2)
    figura_did(app, tela, fonte, "m7_wa_u_depois", alg_poligono="Weiler–Atherton")
    figura_did(app, tela, fonte, "m7_sh_l_concavo", sujeito="região de interesse",
               tipo_janela="L côncavo", alg_poligono="Sutherland–Hodgman", vista_recorte=0)
    figura_did(app, tela, fonte, "m7_wa_l_concavo", alg_poligono="Weiler–Atherton")


if __name__ == "__main__":
    PASTA.mkdir(parents=True, exist_ok=True)
    figuras_visualizador()
    figuras_didatico()
    print(f"\n{len(list(PASTA.glob('*.png')))} figuras em {PASTA}/")
