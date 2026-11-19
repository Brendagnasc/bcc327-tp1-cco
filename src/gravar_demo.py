"""
gravar_demo.py

Grava a demonstração curta que aparece no topo do README (demo/demo.gif e
demo/demo.mp4): crescimento da árvore, fluxo de sangue, clique num vaso,
transparência, filtro por raio e o Ray Tracing da árvore.

Cada cena é uma função que ajusta o Estado do visualizador quadro a quadro;
os quadros são capturados com glReadPixels, recebem uma legenda (Pillow) e
são montados pelo ffmpeg.

Uso (a partir da raiz do repositório; precisa do ffmpeg instalado):
    python src/gravar_demo.py
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pygame
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont

import json
import types

import metricas
import raytracer_arvore
import visualizador as vis
from leitor_vtk import ler_vtk

FPS = 15
LADO = 540                    # tamanho do quadro final (quadrado)
FAIXA = 64                    # faixa da legenda, embaixo
MODELO = "data/tree3D_Nterm256.vtk"
SAIDA = Path("demo")


def carregar_fonte(tamanho):
    for nome in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            pass
    caminho = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if caminho.exists():
        return ImageFont.truetype(str(caminho), tamanho)
    return ImageFont.load_default()


FONTE_TITULO = carregar_fonte(21)
FONTE_SUB = carregar_fonte(15)


def legendar(imagem, titulo, sub=""):
    """Reduz o quadro e acrescenta a faixa com a legenda embaixo."""
    quadro = Image.new("RGB", (LADO, LADO + FAIXA), (20, 35, 60))
    quadro.paste(imagem.resize((LADO, LADO), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(quadro)
    d.text((16, LADO + 9), titulo, font=FONTE_TITULO, fill=(255, 255, 255))
    if sub:
        d.text((16, LADO + 38), sub, font=FONTE_SUB, fill=(190, 205, 225))
    return quadro


def capturar_tela():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    px = glReadPixels(0, 0, vis.LARGURA, vis.ALTURA, GL_RGB, GL_UNSIGNED_BYTE)
    img = np.frombuffer(px, np.uint8).reshape(vis.ALTURA, vis.LARGURA, 3)[::-1]
    return Image.fromarray(img.copy())


def pixel_do_vaso(estado, k):
    """Pixel onde aparece o ponto médio do vaso k (para simular o clique)."""
    a, b = estado.arvore.pontos[estado.arvore.segmentos[k]]
    c = (estado.matriz_projecao() @ estado.matriz_visao() @ estado.matriz_modelo()
         @ np.append((a + b) / 2, 1.0))
    c = c / c[3]
    return (int(round((c[0] + 1) / 2 * vis.LARGURA - 0.5)),
            int(round((1 - c[1]) / 2 * vis.ALTURA - 0.5)))


class Gravador:
    def __init__(self, pasta):
        self.pasta = pasta
        self.n = 0
        self.angulo = 20.0               # rotação contínua entre as cenas

    def salvar(self, imagem):
        imagem.save(self.pasta / f"q{self.n:05d}.png")
        self.n += 1

    def cena(self, estado, segundos, titulo, sub="", ajuste=None, giro=24.0):
        """Grava 'segundos' de animação; ajuste(estado, fração) muda a cena."""
        total = int(segundos * FPS)
        for i in range(total):
            self.angulo += giro / FPS        # graus por segundo
            estado.ang_y = self.angulo
            estado.tempo = self.n / FPS
            if ajuste:
                ajuste(estado, i / max(total - 1, 1))
            vis.desenhar_quadro(estado, None)
            texto_sub = sub(estado) if callable(sub) else sub
            self.salvar(legendar(capturar_tela(), titulo, texto_sub))
            pygame.display.flip()


def escolher_vaso(estado):
    """Vaso que irriga entre 15% e 45% dos terminais (a subárvore se destaca
    sem tomar a árvore toda) e cuja subárvore está mais voltada para a câmera."""
    met, arv = estado.metricas, estado.arvore
    mv = estado.matriz_visao() @ estado.matriz_modelo()
    meio = (arv.pontos[arv.segmentos[:, 0]] + arv.pontos[arv.segmentos[:, 1]]) / 2
    z = meio @ mv[2, :3] + mv[2, 3]                    # maior z = mais perto da câmera
    fracao = met["terminais_abaixo"] / met["n_terminais"]
    ok = (fracao >= 0.15) & (fracao <= 0.45)

    def nota(k):
        return z[metricas.subarvore(met, k) + [k]].mean()

    for k in sorted(np.nonzero(ok)[0], key=nota, reverse=True):
        with contextlib.redirect_stdout(io.StringIO()):
            estado.selecionar(*pixel_do_vaso(estado, k))
        if estado.selecionado == k:
            return k
    return estado.selecionado


def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg não encontrado: instale com 'sudo apt install ffmpeg'")
    SAIDA.mkdir(exist_ok=True)

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((vis.LARGURA, vis.ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gravando demonstração...")
    vis.inicializar_gl()

    arvore = ler_vtk(MODELO)
    estado = vis.Estado(arvore)
    estado.fbo_msaa, estado.amostras_msaa = vis.criar_fbo_msaa(4)
    estado.modo, estado.vista, estado.dist = 1, 3, 3.5
    estado.msaa, estado.eixos, estado.painel, estado.hud = True, False, False, False
    estado.reconstruir()
    n_vasos, n_term = arvore.n_segmentos, estado.metricas["n_terminais"]

    with tempfile.TemporaryDirectory() as tmp:
        g = Gravador(Path(tmp))

        # 0. crescimento, geração por geração (mais rápido que no programa)
        vis.DURACAO_CRESCIMENTO = 4.0
        estado.inicio_crescimento = 0.0
        g.cena(estado, 4.3, "Crescimento da árvore, geração por geração",
               lambda e: (f"geração {min(int(e.nivel_crescimento() or 99), estado.metricas['geracao_max'])}"
                          f" de {estado.metricas['geracao_max']}"))

        # 1. a árvore
        g.cena(estado, 1.5, "Árvore arterial sintética (método CCO)",
               f"{n_vasos} vasos, {n_term} terminais | cor e espessura pelo raio")

        # 2. fluxo
        estado.fluxo = True
        estado.atualizar_cores()
        g.cena(estado, 3.5, "Fluxo de sangue: pulsos saindo da raiz",
               "velocidade de cada vaso = fluxo / área da seção")

        # 3. clique num vaso
        k = escolher_vaso(estado)
        ter = metricas.territorio(arvore, estado.metricas, k)
        sub = (f"amarelo: caminho até a raiz | verde: o que ele irriga "
               f"({100 * ter['fracao_fluxo']:.0f}% do fluxo)")
        g.cena(estado, 3.5, "Clique num vaso (Ray Casting)", sub, giro=6.0)

        # 4. transparência
        estado.selecionado = estado.ponto_selecao = None
        estado.fluxo = False
        estado.opacidade = 2
        estado.atualizar_cores()
        g.cena(estado, 2.5, "Transparência com Painter's Algorithm",
               "triângulos desenhados do mais distante ao mais próximo")

        # 5. filtro por raio: o mínimo sobe até sobrarem os vasos principais
        estado.opacidade = 0
        rmin, alvo = float(arvore.raios.min()), float(np.quantile(arvore.raios, 0.8))

        def filtrar(e, f):
            e.filtro_min = rmin + (alvo - rmin) * min(1.0, 1.4 * f)

        g.cena(estado, 3.0, "Filtro por raio", lambda e: (
            f"raio ≥ {e.filtro_min:.2f} mm: {int(e.visiveis().sum())} de {n_vasos} vasos"
        ).replace(".", ","), ajuste=filtrar)

        # 6. Ray Tracer da árvore, na mesma vista do último quadro, com zoom lento
        estado.filtro_min = rmin
        campos = ["modo", "mapa", "vista", "dist", "ang_x", "ang_y", "ang_z",
                  "tx", "ty", "tz", "s", "exagero", "perspectiva"]
        vista = {c: getattr(estado, c) for c in campos}
        vista["luz"] = [float(x) for x in estado.luz]
        args = types.SimpleNamespace(modelo=MODELO, largura=LADO, rapido=False, ssaa=1,
                                     vista=json.dumps(vista, default=float), tema="escuro")
        img, _, segundos, _, _ = raytracer_arvore.renderizar(args)
        print(f"ray tracing: {segundos:.1f} s")
        rt = Image.fromarray(img)
        for i in range(int(3.0 * FPS)):
            z = 1.0 + 0.10 * i / (3.0 * FPS)
            w = LADO / z
            corte = rt.crop(((LADO - w) / 2, (LADO - w) / 2, (LADO + w) / 2, (LADO + w) / 2))
            g.salvar(legendar(corte, "Ray Tracing da árvore",
                              "sombras entre os vasos e reflexo no piso"))

        pygame.display.quit()
        print(f"{g.n} quadros ({g.n / FPS:.1f} s)")

        entrada = ["-framerate", str(FPS), "-i", str(Path(tmp) / "q%05d.png")]
        # MP4 (H.264)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *entrada, "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                        str(SAIDA / "demo.mp4")], check=True)
        # GIF: paleta de 128 cores calculada para o vídeo inteiro (melhor qualidade)
        filtro = ("fps=9,scale=380:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *entrada, "-vf", filtro,
                        str(SAIDA / "demo.gif")], check=True)

    for nome in ("demo.gif", "demo.mp4"):
        print(f"{SAIDA / nome}: {(SAIDA / nome).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
