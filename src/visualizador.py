"""
visualizador.py

Visualizador principal da árvore CCO (pygame-ce + PyOpenGL).
Versão 5: MSAA ligável (antialiasing da API), tubos 3D com espessura
proporcional ao raio e câmera com matrizes próprias (transformacoes.py).

Uso:
    python src/visualizador.py [arquivo.vtk]

Teclas e mouse:
    arrastar (botão esq.)  rotação em torno de x e y
    roda do mouse          zoom (aproxima / afasta a câmera)
    Z / X                  rotação em torno de z
    setas                  translação em x e y
    PageUp / PageDown      translação em z
    , / .                  escala (diminui / aumenta)
    1 2 3 4                vista frontal / lateral / superior / isométrica
    P                      alterna projeção ortográfica / perspectiva
    A                      mostra / esconde os eixos do mundo
    R                      restaura transformações e vista
    M                      liga / desliga o antialiasing MSAA
    S                      salva a tela em report/figures/
    E                      modo: linhas + cor / espessura + cor / espessura sólida
    C                      mapa de cores (jet / viridis)
    + -                    exagero da espessura
    ESC                    sai
"""

import os
import sys
from pathlib import Path

# No Linux/WSL (WSLg) a variável WAYLAND_DISPLAY existe e o PyOpenGL tentaria usar
# EGL, mas o pygame cria o contexto via X11/GLX. Forçar GLX evita esse conflito.
# Só vale para Linux: no Windows e no macOS o PyOpenGL escolhe sozinho.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pygame
from OpenGL.GL import *

import transformacoes as T
from cores import mapear_cores
from geometria import faixas_2d, tubos_3d
from leitor_vtk import ler_vtk

LARGURA, ALTURA = 900, 900
FUNDO = (1.0, 1.0, 1.0)
VERMELHO = (0.8, 0.1, 0.1)
MODOS = ["linhas + cor", "espessura + cor", "espessura sólida"]
MAPAS = ["jet", "viridis"]
MM_PARA_M = 1e-3   # raio vem em mm; coordenadas em m
FOVY = 45.0        # abertura vertical da câmera perspectiva, em graus
DIST_PADRAO = 3.0  # distância da câmera ao alvo (origem do mundo)

# Vistas do observador: direção de onde a câmera olha e vetor "cima"
VISTAS = [
    ("frontal", (0, 0, 1), (0, 1, 0)),
    ("lateral", (1, 0, 0), (0, 1, 0)),
    ("superior", (0, 1, 0), (0, 0, -1)),
    ("isométrica", (1, 1, 1), (0, 1, 0)),
]


class Estado:
    def __init__(self, arvore):
        self.arvore = arvore
        self.modo = 0
        self.mapa = 0
        self.exagero = 2.0
        self.eixos = True
        self.msaa = False
        self.fbo_msaa = None            # framebuffer multiamostrado (criado em main)
        self.amostras_msaa = 0
        self.vertices = self.cores = self.normais = None
        self.primitiva = GL_LINES

        # Normalização: leva o modelo para perto da origem com tamanho ~1,
        # para a mesma câmera servir aos modelos 2D e 3D
        minimo, maximo = arvore.pontos.min(axis=0), arvore.pontos.max(axis=0)
        centro = (minimo + maximo) / 2
        meio = (maximo - minimo).max() / 2
        self.normalizacao = T.escala(1 / meio, 1 / meio, 1 / meio) @ T.translacao(*(-centro))
        self.restaurar()

    def restaurar(self):
        self.tx = self.ty = self.tz = 0.0
        self.ang_x = self.ang_y = self.ang_z = 0.0
        self.s = 1.0
        self.vista = 0
        self.dist = DIST_PADRAO
        self.perspectiva = True

    # ---------------------------------------------------------------- matrizes
    def matriz_modelo(self):
        """M = T . Rx . Ry . Rz . S . N (lida da direita para a esquerda:
        normaliza, escala, gira e por último translada)."""
        return (T.translacao(self.tx, self.ty, self.tz)
                @ T.rotacao_x(self.ang_x) @ T.rotacao_y(self.ang_y) @ T.rotacao_z(self.ang_z)
                @ T.escala(self.s, self.s, self.s)
                @ self.normalizacao)

    def matriz_visao(self):
        _, direcao, cima = VISTAS[self.vista]
        direcao = np.array(direcao, dtype=float)
        olho = direcao / np.linalg.norm(direcao) * self.dist
        return T.olhar_para(olho, (0, 0, 0), cima)

    def matriz_projecao(self):
        aspecto = LARGURA / ALTURA
        if self.perspectiva:
            return T.perspectiva(FOVY, aspecto, 0.01, 100.0)
        # Caixa ortográfica com a mesma altura que o frustum tem no alvo,
        # para o modelo não "pular" de tamanho ao trocar de projeção
        h = self.dist * np.tan(np.radians(FOVY) / 2)
        return T.ortografica(-h * aspecto, h * aspecto, -h, h, 0.01, 100.0)

    # ---------------------------------------------------------------- geometria
    def reconstruir(self):
        arv = self.arvore
        cores_seg = mapear_cores(arv.raios, arv.raios.min(), arv.raios.max(),
                                 MAPAS[self.mapa])
        if self.modo == 2:
            cores_seg = np.tile(VERMELHO, (arv.n_segmentos, 1))

        raio = arv.raios * MM_PARA_M * self.exagero
        self.normais = None
        if self.modo == 0:                    # linhas
            self.vertices = arv.pontos[arv.segmentos].reshape(-1, 3)
            self.cores = np.repeat(cores_seg, 2, axis=0)
            self.primitiva = GL_LINES
        elif not arv.eh_3d:                   # 2D: faixas planas
            self.vertices, seg_id = faixas_2d(arv.pontos, arv.segmentos, raio)
            self.cores = cores_seg[seg_id]
            self.primitiva = GL_TRIANGLES
        else:                                 # 3D: tubos com normais para iluminação
            self.vertices, self.normais, seg_id = tubos_3d(arv.pontos, arv.segmentos, raio)
            self.cores = cores_seg[seg_id]
            self.primitiva = GL_TRIANGLES

        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.cores = np.ascontiguousarray(self.cores, dtype=np.float32)
        self.enviar_para_gpu()

    def enviar_para_gpu(self):
        """Copia os arrays para buffers do OpenGL (VBOs) uma única vez.
        Sem isso, os ~400 mil vértices dos tubos seriam reenviados a cada quadro."""
        if not hasattr(self, "vbos"):
            self.vbos = glGenBuffers(3)
        for vbo, dados in zip(self.vbos, (self.vertices, self.cores, self.normais)):
            if dados is not None:
                dados = np.ascontiguousarray(dados, dtype=np.float32)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, dados.nbytes, dados, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def descricao(self):
        proj = "perspectiva" if self.perspectiva else "ortográfica"
        if self.amostras_msaa == 0:
            aa = "MSAA indisponível"
        else:
            aa = f"MSAA {self.amostras_msaa}x ligado" if self.msaa else "MSAA desligado"
        return (f"{self.arvore.nome} | {proj} | vista {VISTAS[self.vista][0]} | "
                f"{MODOS[self.modo]} | exagero {self.exagero:.1f}x | {aa}")


# -------------------------------------------------------------------- desenho
def desenhar_arvore(estado):
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(T.para_opengl(estado.matriz_visao() @ estado.matriz_modelo()))

    vbo_v, vbo_c, vbo_n = estado.vbos
    iluminar = estado.normais is not None
    if iluminar:
        glEnable(GL_LIGHTING)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_n)
        glNormalPointer(GL_FLOAT, 0, None)     # None = começa no início do buffer

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
    glColorPointer(3, GL_FLOAT, 0, None)
    glDrawArrays(estado.primitiva, 0, len(estado.vertices))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    if iluminar:
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_LIGHTING)


def configurar_iluminacao():
    """Iluminação da API (OpenGL fixo), usada só para dar volume aos tubos.
    Uma luz direcional + luz ambiente; o estudo de iluminação é tema do TP2."""
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.6, 0.6, 0.6, 1.0))
    glEnable(GL_COLOR_MATERIAL)                     # a cor do vértice vira o material
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_NORMALIZE)                          # a matriz M tem escala: renormaliza
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)


def desenhar_eixos(estado):
    """Eixos do mundo (x vermelho, y verde, z azul). Usam só a matriz de visão,
    então ficam parados enquanto o modelo é transformado."""
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(T.para_opengl(estado.matriz_visao()))
    glLineWidth(3.0)
    glBegin(GL_LINES)
    for eixo, cor in zip(np.identity(3), np.identity(3) * 0.8):
        glColor3f(*cor)
        glVertex3f(0, 0, 0)
        glVertex3f(*(eixo * 1.3))
    glEnd()
    glLineWidth(2.0)


def desenhar_texto(fonte, texto, x, y):
    """Escreve texto na tela em coordenadas de pixel (origem embaixo à esquerda)."""
    superficie = fonte.render(texto, True, (20, 20, 20))
    dados = pygame.image.tobytes(superficie, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(superficie.get_width(), superficie.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, dados)


def desenhar_barra_cores(estado, fonte):
    """Barra horizontal no topo da tela, desenhada em coordenadas de pixel."""
    arv = estado.arvore
    x0, x1, y0, y1 = LARGURA * 0.3, LARGURA * 0.7, ALTURA - 60, ALTURA - 40
    passos = 64
    valores = np.linspace(arv.raios.min(), arv.raios.max(), passos + 1)
    cores = mapear_cores(valores, valores[0], valores[-1], MAPAS[estado.mapa])
    xs = np.linspace(x0, x1, passos + 1)

    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(T.para_opengl(T.ortografica(0, LARGURA, 0, ALTURA, -1, 1)))
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glBegin(GL_QUAD_STRIP)
    for x, cor in zip(xs, cores):
        glColor3f(*cor)
        glVertex2f(x, y0)
        glVertex2f(x, y1)
    glEnd()

    desenhar_texto(fonte, f"{arv.raios.min():.3f}", x0 - 55, y0 + 2)
    desenhar_texto(fonte, f"{arv.raios.max():.3f}", x1 + 8, y0 + 2)
    desenhar_texto(fonte, "raio (mm)", LARGURA / 2 - 35, y1 + 6)


def criar_fbo_msaa(amostras=4):
    """Cria um framebuffer fora da tela (FBO) com várias amostras por pixel.

    MSAA: cada pixel guarda 'amostras' valores de cor e profundidade, um por
    posição de amostra. O teste de cobertura e de profundidade é feito por
    amostra, mas o fragmento é sombreado uma vez só por pixel (mais barato
    que o supersampling). No final, glBlitFramebuffer faz a "resolução":
    a média das amostras de cada pixel vai para a janela.
    """
    try:
        amostras = min(amostras, glGetIntegerv(GL_MAX_SAMPLES))
        fbo = glGenFramebuffers(1)
        cor, prof = glGenRenderbuffers(2)
        glBindRenderbuffer(GL_RENDERBUFFER, cor)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, amostras, GL_RGBA8, LARGURA, ALTURA)
        glBindRenderbuffer(GL_RENDERBUFFER, prof)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, amostras, GL_DEPTH_COMPONENT24,
                                         LARGURA, ALTURA)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, cor)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, prof)
        completo = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        return (fbo, amostras) if completo else (None, 0)
    except Exception:
        return None, 0


def desenhar_quadro(estado, fonte):
    usar_msaa = estado.msaa and estado.fbo_msaa is not None
    if usar_msaa:                              # desenha a cena no FBO multiamostrado
        glBindFramebuffer(GL_FRAMEBUFFER, estado.fbo_msaa)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)                   # cena 3D com teste de profundidade
    glMatrixMode(GL_PROJECTION)               # projeção vale para eixos e árvore
    glLoadMatrixf(T.para_opengl(estado.matriz_projecao()))

    # Luz definida com a modelview identidade: fica presa à câmera ("lanterna")
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 0.6, 1.0, 0.0))
    if estado.eixos:
        desenhar_eixos(estado)
    desenhar_arvore(estado)

    if usar_msaa:                              # resolve: média das amostras -> janela
        glBindFramebuffer(GL_READ_FRAMEBUFFER, estado.fbo_msaa)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, LARGURA, ALTURA, 0, 0, LARGURA, ALTURA,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    glDisable(GL_DEPTH_TEST)                  # sobreposições 2D sempre por cima
    if estado.modo != 2:
        desenhar_barra_cores(estado, fonte)
    desenhar_texto(fonte, estado.descricao(), 10, 10)


# -------------------------------------------------------------------- entrada
def tratar_tecla(estado, tecla):
    passo = 0.05
    if tecla == pygame.K_e:
        estado.modo = (estado.modo + 1) % len(MODOS)
        estado.reconstruir()
    elif tecla == pygame.K_c:
        estado.mapa = (estado.mapa + 1) % len(MAPAS)
        estado.reconstruir()
    elif tecla in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        estado.exagero = min(estado.exagero + 0.5, 10.0)
        estado.reconstruir()
    elif tecla in (pygame.K_MINUS, pygame.K_KP_MINUS):
        estado.exagero = max(estado.exagero - 0.5, 0.5)
        estado.reconstruir()
    elif tecla == pygame.K_LEFT:
        estado.tx -= passo
    elif tecla == pygame.K_RIGHT:
        estado.tx += passo
    elif tecla == pygame.K_UP:
        estado.ty += passo
    elif tecla == pygame.K_DOWN:
        estado.ty -= passo
    elif tecla == pygame.K_PAGEUP:
        estado.tz += passo
    elif tecla == pygame.K_PAGEDOWN:
        estado.tz -= passo
    elif tecla == pygame.K_z:
        estado.ang_z += 5
    elif tecla == pygame.K_x:
        estado.ang_z -= 5
    elif tecla == pygame.K_PERIOD:
        estado.s *= 1.1
    elif tecla == pygame.K_COMMA:
        estado.s /= 1.1
    elif tecla in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
        estado.vista = tecla - pygame.K_1
    elif tecla == pygame.K_p:
        estado.perspectiva = not estado.perspectiva
    elif tecla == pygame.K_a:
        estado.eixos = not estado.eixos
    elif tecla == pygame.K_r:
        estado.restaurar()
    elif tecla == pygame.K_m and estado.amostras_msaa > 0:
        estado.msaa = not estado.msaa


def inicializar_gl():
    """Estado inicial do OpenGL (usado pelo main e pelo gerador de figuras)."""
    glClearColor(*FUNDO, 1.0)
    glLineWidth(2.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    configurar_iluminacao()


def capturar(caminho):
    """Lê o framebuffer da janela e salva em PNG (chamar antes do flip)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    dados = glReadPixels(0, 0, LARGURA, ALTURA, GL_RGB, GL_UNSIGNED_BYTE)
    # o OpenGL lê de baixo para cima; flipped=True inverte para a imagem
    imagem = pygame.image.frombytes(bytes(dados), (LARGURA, ALTURA), "RGB", True)
    pygame.image.save(imagem, str(caminho))
    print("salvo:", caminho)


def nome_captura(estado):
    proj = "persp" if estado.perspectiva else "orto"
    vista = ["frontal", "lateral", "superior", "isometrica"][estado.vista]
    modo = ["linhas", "espessura", "solido"][estado.modo]
    aa = "_msaa" if estado.msaa else ""
    return f"report/figures/vis_{estado.arvore.nome}_{modo}_{vista}_{proj}{aa}.png"


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree2D_Nterm064.vtk"
    arvore = ler_vtk(caminho)

    pygame.display.init()             # só vídeo e fonte (evita iniciar o áudio)
    pygame.font.init()
    pygame.display.set_mode((LARGURA, ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption(f"TP1 CCO - {arvore.nome}")
    fonte = pygame.font.Font(None, 24)

    inicializar_gl()

    estado = Estado(arvore)
    estado.fbo_msaa, estado.amostras_msaa = criar_fbo_msaa(4)
    estado.reconstruir()

    relogio = pygame.time.Clock()
    rodando = True
    while rodando:
        salvar = False
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key == pygame.K_s:
                    salvar = True
                else:
                    tratar_tecla(estado, evento.key)
            elif evento.type == pygame.MOUSEMOTION and evento.buttons[0]:
                estado.ang_y += evento.rel[0] * 0.5
                estado.ang_x += evento.rel[1] * 0.5
            elif evento.type == pygame.MOUSEWHEEL:
                fator = 0.9 if evento.y > 0 else 1 / 0.9
                estado.dist = float(np.clip(estado.dist * fator, 0.5, 20.0))

        desenhar_quadro(estado, fonte)
        if salvar:
            capturar(nome_captura(estado))
        pygame.display.flip()
        relogio.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
