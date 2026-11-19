"""
gravar_reels.py

Grava um vídeo vertical curto (1080 x 1920, 30 quadros/s) para Reels/Stories:
a árvore no centro, sobre o código-fonte do próprio projeto rolando ao fundo.
Cada cena mostra uma técnica e, atrás dela, o trecho de código que a implementa.

Fundo transparente da árvore: cada quadro é desenhado DUAS vezes, com o fundo
preto e com o fundo branco. Num pixel com opacidade a e cor c, as leituras são
    preto  = a c          branco = a c + (1 - a)
então  a = 1 - (branco - preto)  e  a c = preto.  Isso vale também para a
neblina, a transparência e as bordas suavizadas pelo MSAA, que viram
semitransparência de verdade por cima do código.

Uso (a partir da raiz do repositório; precisa do ffmpeg instalado):
    python src/gravar_reels.py [saida.mp4]
"""

import contextlib
import io
import json
import keyword
import re
import os
import shutil
import subprocess
import sys
import tokenize
import types
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
    # No WSL, o driver D3D12 (GPU) devolve só o fundo no glReadPixels; a
    # renderização por software (llvmpipe) é mais lenta, mas captura a cena inteira.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import pygame
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont

import metricas
import raytracer_arvore
import visualizador as vis
from leitor_vtk import ler_vtk

FPS = 30
W, H = 1080, 1920              # quadro final (vertical, 9:16)
TAM = 1080                     # lado da imagem da árvore
Y_ARVORE = 430                 # topo da árvore no quadro
MODELO = "data/tree3D_Nterm256.vtk"
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/reels.mp4")
SRC = Path(__file__).parent

# --------------------------------------------------------------- fontes
def fonte(nomes, tamanho):
    for nome in nomes:
        for pasta in ("", "/usr/share/fonts/truetype/dejavu/"):
            try:
                return ImageFont.truetype(pasta + nome, tamanho)
            except OSError:
                pass
    return ImageFont.load_default()


F_TITULO = fonte(["DejaVuSans-Bold.ttf"], 62)
F_SUB = fonte(["DejaVuSans.ttf"], 34)
F_TAG = fonte(["DejaVuSans-Bold.ttf"], 28)
F_CODIGO = fonte(["DejaVuSansMono.ttf"], 25)
F_ARQUIVO = fonte(["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"], 30)
ALTURA_LINHA = 36
LARGURA_CHAR = F_CODIGO.getlength("M")

# --------------------------------------------------------------- código ao fundo
# cores no estilo dos editores escuros
COR = {"padrao": (171, 178, 191), "keyword": (198, 120, 221), "string": (152, 195, 121),
       "comentario": (110, 118, 135), "numero": (209, 154, 102), "funcao": (97, 175, 239),
       "self": (224, 108, 117), "builtin": (86, 182, 194), "num_linha": (70, 76, 92)}
FUNDO_CODIGO = (14, 16, 22)
BUILTINS = {"len", "range", "print", "min", "max", "abs", "float", "int", "zip",
            "enumerate", "sorted", "sum", "np", "isinstance", "list", "dict", "set"}
GLSL_KEYWORDS = {"uniform", "in", "out", "flat", "void", "return", "if", "else", "for",
                 "layout", "location", "const", "discard", "core"}
GLSL_TIPOS = {"float", "int", "bool", "vec2", "vec3", "vec4", "ivec2", "mat3", "mat4",
              "sampler2D", "sampler2DShadow"}
TOKEN_GLSL = re.compile(r"//.*|#\w+|\d+\.\d*|\d+|[A-Za-z_]\w*")


def colorir_glsl(linha, cores, ini, fim):
    """Destaque de GLSL dentro das strings de shader do shaders.py."""
    for m in TOKEN_GLSL.finditer(linha, ini, fim):
        t = m.group()
        if t.startswith("//"):
            cor = COR["comentario"]
        elif t.startswith("#") or t in GLSL_KEYWORDS:
            cor = COR["keyword"]
        elif t[0].isdigit():
            cor = COR["numero"]
        elif t in GLSL_TIPOS:
            cor = COR["builtin"]
        elif linha[m.end():m.end() + 1] == "(":
            cor = COR["funcao"]
        else:
            cor = COR["padrao"]
        for c in range(m.start(), m.end()):
            cores[c] = cor


def colorir(caminho):
    """Lista de linhas; cada linha é uma lista de (coluna, texto, cor)."""
    texto = caminho.read_text(encoding="utf-8")
    linhas = texto.split("\n")
    cores = [[COR["padrao"]] * len(l) for l in linhas]
    anterior = None
    for tok in tokenize.generate_tokens(io.StringIO(texto).readline):
        tipo, valor, (l0, c0), (l1, c1), _ = tok
        if tipo == tokenize.COMMENT:
            cor = COR["comentario"]
        elif tipo == tokenize.STRING and anterior in ("=", "+") and l1 > l0:
            # string de várias linhas atribuída a uma variável: código GLSL
            for l in range(l0, l1 + 1):
                ini = c0 if l == l0 else 0
                fim = c1 if l == l1 else len(linhas[l - 1])
                colorir_glsl(linhas[l - 1], cores[l - 1], ini, fim)
            anterior = valor
            continue
        elif tipo == tokenize.STRING:
            cor = COR["string"]
        elif tipo == tokenize.NUMBER:
            cor = COR["numero"]
        elif tipo == tokenize.NAME:
            if anterior in ("def", "class"):
                cor = COR["funcao"]
            elif keyword.iskeyword(valor):
                cor = COR["keyword"]
            elif valor == "self":
                cor = COR["self"]
            elif valor in BUILTINS:
                cor = COR["builtin"]
            else:
                cor = None
        else:
            cor = None
        if tipo not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            anterior = valor
        if cor is None:
            continue
        for l in range(l0, l1 + 1):                 # tokens de várias linhas (docstrings)
            ini = c0 if l == l0 else 0
            fim = c1 if l == l1 else len(linhas[l - 1])
            for c in range(ini, min(fim, len(cores[l - 1]))):
                cores[l - 1][c] = cor
    resultado = []
    for linha, cs in zip(linhas, cores):
        trechos, i = [], 0
        while i < len(linha):
            j = i
            while j < len(linha) and cs[j] == cs[i]:
                j += 1
            trechos.append((i, linha[i:j], cs[i]))
            i = j
        resultado.append(trechos)
    return resultado


def imagem_codigo(arquivo, ancora, altura):
    """Imagem do código a partir de algumas linhas antes da 'ancora'."""
    linhas = colorir(SRC / arquivo)
    texto = (SRC / arquivo).read_text(encoding="utf-8").split("\n")
    inicio = next(i for i, l in enumerate(texto) if ancora in l)
    inicio = max(0, inicio - 14)       # a âncora aparece logo abaixo dos títulos
    n = altura // ALTURA_LINHA + 2
    img = Image.new("RGB", (W, n * ALTURA_LINHA), FUNDO_CODIGO)
    d = ImageDraw.Draw(img)
    x0 = 26 + 4 * LARGURA_CHAR
    for k in range(n):
        i = inicio + k
        if i >= len(linhas):
            break
        y = k * ALTURA_LINHA + 6
        d.text((26, y), f"{i + 1:>3}", font=F_CODIGO, fill=COR["num_linha"])
        for col, trecho, cor in linhas[i]:
            d.text((x0 + col * LARGURA_CHAR, y), trecho, font=F_CODIGO, fill=cor)
    return np.asarray(img, dtype=np.float32)


def mascara_fundo():
    """Escurece o código onde há texto e atrás da árvore, para tudo ficar legível."""
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    topo = np.clip((560 - y) / 200, 0, 1) * 0.85                  # faixa dos títulos
    base = np.clip((y - 1430) / 220, 0, 1) * 0.55                 # faixa do arquivo
    cy = Y_ARVORE + TAM / 2
    r = np.sqrt(((x - W / 2) / (W * 0.55)) ** 2 + ((y - cy) / (TAM * 0.55)) ** 2)
    centro = np.clip(1 - r, 0, 1) * 0.35                          # atrás da árvore
    escuro = np.maximum(np.maximum(topo, base), centro)
    return (0.62 * (1 - escuro))[..., None]                       # brilho do código


# --------------------------------------------------------------- árvore com alfa
TEMA_PRETO = dict(vis.TEMAS["escuro"], fundo=(0, 0, 0), fundo_alto=(0, 0, 0), fundo_baixo=(0, 0, 0))
TEMA_BRANCO = dict(vis.TEMAS["escuro"], fundo=(1, 1, 1), fundo_alto=(1, 1, 1), fundo_baixo=(1, 1, 1))


def ler_tela():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    px = glReadPixels(0, 0, TAM, TAM, GL_RGB, GL_UNSIGNED_BYTE)
    return np.frombuffer(px, np.uint8).reshape(TAM, TAM, 3)[::-1].astype(np.float32)


def arvore_rgba(estado, fonte_gl):
    """(cor pré-multiplicada, alfa) da cena, com o fundo transparente."""
    vis.tema = TEMA_PRETO
    vis.desenhar_quadro(estado, fonte_gl)
    preto = ler_tela()
    vis.tema = TEMA_BRANCO
    vis.desenhar_quadro(estado, fonte_gl)
    branco = ler_tela()
    vis.tema = vis.TEMAS["escuro"]
    pygame.display.flip()
    alfa = np.clip(1 - (branco - preto).mean(axis=2, keepdims=True) / 255, 0, 1)
    return preto, alfa


# --------------------------------------------------------------- textos
def texto_centralizado(d, y, texto, f, cor, sombra=True):
    w = d.textlength(texto, font=f)
    x = (W - w) / 2
    if sombra:
        d.text((x + 2, y + 3), texto, font=f, fill=(0, 0, 0))
    d.text((x, y), texto, font=f, fill=cor)


def camada_textos(titulo, sub, arquivo, funcao):
    """Camada RGBA (Pillow) com os textos da cena."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    texto_centralizado(d, 175, "PYTHON + OPENGL  ·  DO ZERO", F_TAG, (255, 196, 70))
    texto_centralizado(d, 235, titulo, F_TITULO, (255, 255, 255))
    for i, linha in enumerate(sub.split("\n")):
        texto_centralizado(d, 322 + 44 * i, linha, F_SUB, (200, 212, 232))
    if arquivo:
        rotulo = f"{arquivo}  ›  {funcao}"
        w = d.textlength(rotulo, font=F_ARQUIVO)
        x, y = (W - w) / 2, 1540
        d.rounded_rectangle((x - 22, y - 14, x + w + 22, y + 46), radius=16,
                            fill=(20, 24, 34, 230), outline=(97, 175, 239, 255), width=2)
        d.text((x, y), rotulo, font=F_ARQUIVO, fill=(220, 228, 240))
    rgba = np.asarray(img, dtype=np.float32) / 255
    alfa = np.ascontiguousarray(rgba[..., 3:4])
    return np.ascontiguousarray(rgba[..., :3] * 255 * alfa), alfa   # cor pré-multiplicada


# --------------------------------------------------------------- gravação
class Reels:
    def __init__(self, ffmpeg, fonte_gl):
        self.ffmpeg = ffmpeg
        self.fonte_gl = fonte_gl
        self.n = 0
        self.angulo = 20.0
        self.brilho = mascara_fundo()
        self.codigo_anterior = None

    def quadro(self, codigo, arvore, textos, alfa_textos):
        """Junta código (fundo), árvore (pré-multiplicada + alfa) e textos."""
        img = codigo * self.brilho
        if arvore is not None:
            cor, alfa = arvore
            regiao = img[Y_ARVORE:Y_ARVORE + TAM]
            img[Y_ARVORE:Y_ARVORE + TAM] = cor + (1 - alfa) * regiao
        cor_txt, a = textos
        img = img * (1 - a * alfa_textos) + cor_txt * alfa_textos
        self.ffmpeg.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
        self.n += 1

    def cena(self, estado, segundos, titulo, sub, arquivo, ancora, funcao,
             ajuste=None, giro=20.0, imagem=None):
        """Grava uma cena. 'imagem(fração)' substitui a árvore (Ray Tracing)."""
        total = int(segundos * FPS)
        rolagem = 26.0                                     # pixels por segundo
        cod = imagem_codigo(arquivo, ancora, H + int(rolagem * segundos) + 40)
        estatico = camada_textos(titulo, "" if callable(sub) else sub, arquivo, funcao)
        for i in range(total):
            f = i / max(total - 1, 1)
            t = i / FPS
            y = int(rolagem * t)
            codigo = cod[y:y + H].copy()
            if self.codigo_anterior is not None and t < 0.4:   # transição do código
                m = t / 0.4
                codigo = (1 - m) * self.codigo_anterior + m * codigo
            if imagem is None:
                self.angulo += giro / FPS
                estado.ang_y = self.angulo
                estado.tempo = self.n / FPS
                if ajuste:
                    ajuste(estado, f)
                arvore = arvore_rgba(estado, self.fonte_gl)
            else:
                arvore = imagem(f)
            textos = camada_textos(titulo, sub(estado), arquivo, funcao) if callable(sub) else estatico
            entrada = min(1.0, t / 0.35)                   # textos surgem
            self.quadro(codigo, arvore, textos, entrada)
        self.codigo_anterior = codigo


def escolher_vaso(estado):
    """Vaso que irriga entre 15% e 45% dos terminais, com a subárvore voltada
    para a câmera (o mesmo critério do gravar_demo.py)."""
    met, arv = estado.metricas, estado.arvore
    mv = estado.matriz_visao() @ estado.matriz_modelo()
    meio = (arv.pontos[arv.segmentos[:, 0]] + arv.pontos[arv.segmentos[:, 1]]) / 2
    z = meio @ mv[2, :3] + mv[2, 3]
    fracao = met["terminais_abaixo"] / met["n_terminais"]
    candidatos = np.nonzero((fracao >= 0.15) & (fracao <= 0.45))[0]
    for k in sorted(candidatos, key=lambda k: z[metricas.subarvore(met, k) + [k]].mean(),
                    reverse=True):
        a, b = arv.pontos[arv.segmentos[k]]
        c = estado.matriz_projecao() @ mv @ np.append((a + b) / 2, 1.0)
        c = c / c[3]
        px = (int(round((c[0] + 1) / 2 * TAM - 0.5)), int(round((1 - c[1]) / 2 * TAM - 0.5)))
        with contextlib.redirect_stdout(io.StringIO()):
            estado.selecionar(*px)
        if estado.selecionado == k:
            return k
    return estado.selecionado


def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg não encontrado: instale com 'sudo apt install ffmpeg'")
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    vis.LARGURA = vis.ALTURA = TAM
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((TAM, TAM), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Gravando o vídeo vertical...")
    vis.inicializar_gl()
    vis.desenhar_luz = lambda estado: None        # sem o marcador da luz no vídeo
    fonte_gl = pygame.font.Font(None, 40)

    arvore = ler_vtk(MODELO)
    estado = vis.Estado(arvore)
    estado.fbo_msaa, estado.amostras_msaa = vis.criar_fbo_msaa(4)
    estado.modo, estado.vista, estado.dist = 1, 3, 2.9
    estado.msaa, estado.eixos, estado.painel, estado.hud = True, False, False, False
    estado.plano, estado.sombra, estado.neblina = False, False, True
    estado.reconstruir()
    n_vasos = arvore.n_segmentos
    g_max = estado.metricas["geracao_max"]

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
         str(SAIDA)], stdin=subprocess.PIPE)
    r = Reels(ffmpeg, fonte_gl)

    # 1. crescimento
    vis.DURACAO_CRESCIMENTO = 3.2
    estado.inicio_crescimento = 0.0
    r.cena(estado, 3.6, "Vasos sanguíneos em 3D",
           lambda e: f"crescendo geração por geração: "
                     f"{min(int(e.nivel_crescimento() or 99), g_max)} de {g_max}",
           "metricas.py", "fila = deque", "busca em largura")

    # 2. fluxo
    estado.fluxo = True
    estado.atualizar_cores()
    r.cena(estado, 3.6, "Fluxo de sangue", "pulsos saindo da raiz a 72 batimentos/min\n"
           "velocidade = fluxo / área de cada vaso",
           "metricas.py", "def tempos_de_transito", "tempos_de_transito()")

    # 3. cor por geração
    estado.fluxo = False
    estado.mapa = 2
    estado.reconstruir()
    r.cena(estado, 2.6, "Cor por geração", "da raiz (roxo) até as pontas (amarelo)",
           "cores.py", "def mapear_cores", "mapear_cores()")

    # 4. Flat x Phong, com a divisória passeando (de perto, para ver as facetas)
    estado.mapa = 0
    estado.reconstruir()
    estado.comparacao = 0
    dist0 = estado.dist

    def comparar(e, f):
        e.divisoria = 0.5 + 0.32 * np.sin(2 * np.pi * f)
        e.dist = dist0 - 0.5 * np.sin(np.pi * min(1.0, f * 1.6) / 2)

    r.cena(estado, 3.6, "Flat × Phong", "a mesma luz calculada por triângulo\nou por pixel (shaders GLSL)",
           "shaders.py", "vec3 iluminar", "iluminar() em GLSL", ajuste=comparar, giro=8.0)
    estado.comparacao, estado.dist = None, dist0

    # 5. clique num vaso
    k = escolher_vaso(estado)
    ter = metricas.territorio(arvore, estado.metricas, k)
    r.cena(estado, 3.6, "Clique num vaso", f"Ray Casting: amarelo = caminho até a raiz\n"
           f"verde = o que ele irriga ({100 * ter['fracao_fluxo']:.0f}% do fluxo)",
           "raycasting.py", "def intersecao_cilindros", "intersecao_cilindros()", giro=5.0)

    # 6. transparência
    estado.selecionado = estado.ponto_selecao = None
    estado.opacidade = 2
    estado.atualizar_cores()
    r.cena(estado, 2.8, "Transparência", "Painter's Algorithm: do mais distante\nao mais próximo",
           "visualizador.py", "def ordenar_tras_para_frente", "ordenar_tras_para_frente()")

    # 7. filtro por raio
    estado.opacidade = 0
    rmin, alvo = float(arvore.raios.min()), float(np.quantile(arvore.raios, 0.8))

    def filtrar(e, f):
        e.filtro_min = rmin + (alvo - rmin) * min(1.0, 1.4 * f)

    r.cena(estado, 2.8, "Filtro por raio", lambda e: (
        f"raio ≥ {e.filtro_min:.2f} mm: {int(e.visiveis().sum())} de {n_vasos} vasos").replace(".", ","),
        "visualizador.py", "def visiveis", "visiveis()", ajuste=filtrar)

    # 8. Ray Tracing da árvore, na vista do último quadro
    estado.filtro_min = rmin
    campos = ["modo", "mapa", "vista", "dist", "ang_x", "ang_y", "ang_z",
              "tx", "ty", "tz", "s", "exagero", "perspectiva"]
    vista = {c: getattr(estado, c) for c in campos}
    vista["luz"] = [float(x) for x in estado.luz]
    args = types.SimpleNamespace(modelo=MODELO, largura=TAM, rapido=False, ssaa=1,
                                 vista=json.dumps(vista, default=float), tema="escuro")
    print("calculando o Ray Tracing...", flush=True)
    rt, _, segundos, _, _ = raytracer_arvore.renderizar(args)
    print(f"ray tracing: {segundos:.0f} s", flush=True)
    rt = Image.fromarray(rt)
    ultimo = arvore_rgba(estado, fonte_gl)
    # bordas suaves: a imagem do Ray Tracing se funde ao código em vez de virar um quadrado
    d = np.minimum(np.arange(TAM), np.arange(TAM)[::-1]).astype(np.float32)
    borda = np.clip(d / 140, 0, 1)
    pena = (np.minimum.outer(borda, borda) ** 1.5)[..., None]

    def imagem_rt(f):
        z = 1.0 + 0.08 * f
        w = TAM / z
        corte = rt.crop(((TAM - w) / 2, (TAM - w) / 2, (TAM + w) / 2, (TAM + w) / 2))
        cor = np.asarray(corte.resize((TAM, TAM), Image.LANCZOS), dtype=np.float32)
        m = min(1.0, f * 3.2 / 0.5)                         # 0,5 s de transição
        alfa = ultimo[1] * (1 - m) + pena * m
        return ultimo[0] * (1 - m) + cor * pena * m, alfa

    r.cena(estado, 3.2, "Ray Tracing da árvore", "sombras entre os vasos e reflexo no piso",
           "raytracer_arvore.py", "def tracar", "Tracador.tracar()", imagem=imagem_rt)

    # 9. encerramento
    estado.dist = 3.2
    r.cena(estado, 2.6, "Python · OpenGL · NumPy", "shaders, Ray Casting e Ray Tracing\nescritos do zero",
           "visualizador.py", "def desenhar_quadro", "desenhar_quadro()", giro=30.0)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    pygame.display.quit()
    print(f"{r.n} quadros ({r.n / FPS:.1f} s) -> {SAIDA} "
          f"({SAIDA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
