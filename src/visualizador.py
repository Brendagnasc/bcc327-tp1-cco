"""
visualizador.py

Visualizador principal da árvore CCO (pygame-ce + PyOpenGL).
TP2, versão 4: transparência com alpha blending e Painter's Algorithm,
remoção de superfícies escondidas (Back-Face Culling próprio e do OpenGL,
Z-Buffer), sombreamento Flat / Gouraud / Phong e luz pontual móvel.

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
    G                      sombreamento: Flat / Gouraud / Phong
    K                      componentes: ambiente / + difusa / + especular
    L                      modo luz: setas e PageUp/PageDown movem a luz,
                           + e - mudam a intensidade (L de novo volta ao normal)
    B                      Back-Face Culling: desligado / OpenGL / próprio (N . V)
    D                      liga / desliga o Z-Buffer (teste de profundidade)
    V                      mostra o conteúdo do Z-Buffer em tons de cinza
    F                      pinta de vermelho as faces de trás
    O                      opacidade dos vasos: 1,0 / 0,6 / 0,3
    N                      ordem dos transparentes: Painter's (trás -> frente) / sem ordenação
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

import shaders
import transformacoes as T
from cores import mapear_cores
from geometria import faixas_2d, normais_de_face, tubos_3d
from leitor_vtk import ler_vtk

LARGURA, ALTURA = 900, 900
FUNDO = (1.0, 1.0, 1.0)
VERMELHO = (0.8, 0.1, 0.1)
MODOS = ["linhas + cor", "espessura + cor", "espessura sólida"]
MAPAS = ["jet", "viridis"]
MM_PARA_M = 1e-3   # raio vem em mm; coordenadas em m
LUZ_PADRAO = (1.2, 1.6, 2.0)   # posição inicial da luz pontual (espaço do mundo)
PERTO, LONGE = 0.01, 100.0     # planos near e far das projeções
CULLING = ["desligado", "OpenGL", "próprio (N·V)"]
OPACIDADES = [1.0, 0.6, 0.3]
ORDENS = ["Painter's (trás → frente)", "sem ordenação"]
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
        self.vertices = self.cores = self.normais = self.normais_face = None
        self.sombreamento = "Phong"
        self.componentes = 2            # 0 ambiente, 1 + difusa, 2 + especular
        self.modo_luz = False
        self.luz = np.array(LUZ_PADRAO)  # posição da luz no espaço do MUNDO
        self.intensidade = 1.0
        self.culling = 0
        self.zbuffer = True
        self.ver_profundidade = False
        self.destacar_tras = False
        self.tri_desenhados = 0
        self.opacidade = 0
        self.ordem = 0
        self.primitiva = GL_LINES

        # Normalização: leva o modelo para perto da origem com tamanho ~1,
        # para a mesma câmera servir aos modelos 2D e 3D
        minimo, maximo = arvore.pontos.min(axis=0), arvore.pontos.max(axis=0)
        centro = (minimo + maximo) / 2
        meio = (maximo - minimo).max() / 2
        self.normalizacao = T.escala(1 / meio, 1 / meio, 1 / meio) @ T.translacao(*(-centro))
        self.restaurar()

    def restaurar(self):
        if hasattr(self, "luz"):
            self.luz = np.array(LUZ_PADRAO)
            self.intensidade = 1.0
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
            return T.perspectiva(FOVY, aspecto, PERTO, LONGE)
        # Caixa ortográfica com a mesma altura que o frustum tem no alvo,
        # para o modelo não "pular" de tamanho ao trocar de projeção
        h = self.dist * np.tan(np.radians(FOVY) / 2)
        return T.ortografica(-h * aspecto, h * aspecto, -h, h, PERTO, LONGE)

    # ---------------------------------------------------------------- geometria
    def reconstruir(self):
        arv = self.arvore
        cores_seg = mapear_cores(arv.raios, arv.raios.min(), arv.raios.max(),
                                 MAPAS[self.mapa])
        if self.modo == 2:
            cores_seg = np.tile(VERMELHO, (arv.n_segmentos, 1))

        raio = arv.raios * MM_PARA_M * self.exagero
        self.normais = self.normais_face = None
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
            self.normais_face = normais_de_face(self.vertices, self.normais)
            self.cores = cores_seg[seg_id]
            # dados por triângulo para o culling próprio: normal e centroide
            self.tri_normal = self.normais_face[::3].astype(float)
            self.tri_centro = self.vertices.reshape(-1, 3, 3).mean(axis=1).astype(float)
            self.primitiva = GL_TRIANGLES

        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.cores = np.ascontiguousarray(self.cores, dtype=np.float32)
        self.enviar_para_gpu()

    def enviar_para_gpu(self):
        """Copia os arrays para buffers do OpenGL (VBOs) uma única vez.
        Sem isso, os ~400 mil vértices dos tubos seriam reenviados a cada quadro."""
        if not hasattr(self, "vbos"):
            self.vbos = glGenBuffers(4)     # posição, cor, normal do vértice, normal da face
            self.ebo = glGenBuffers(1)      # índices dos triângulos visíveis (culling próprio)
        for vbo, dados in zip(self.vbos, (self.vertices, self.cores, self.normais,
                                          self.normais_face)):
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
        texto = (f"{self.arvore.nome} | {proj} | vista {VISTAS[self.vista][0]} | "
                 f"{MODOS[self.modo]} | {aa}")
        if self.normais is not None:
            total = len(self.vertices) // 3
            texto += (f"\nculling {CULLING[self.culling]}: {self.tri_desenhados}/{total} "
                      f"triângulos | Z-Buffer {'ligado' if self.zbuffer else 'desligado'}")
            if OPACIDADES[self.opacidade] < 1.0:
                texto += (f" | opacidade {OPACIDADES[self.opacidade]:.1f}".replace(".", ",")
                          + f" | ordem: {ORDENS[self.ordem]}")
            texto += (f"\n{self.sombreamento} | {COMPONENTES[self.componentes]} | "
                      f"luz ({self.luz[0]:.1f}, {self.luz[1]:.1f}, {self.luz[2]:.1f}) "
                      f"x{self.intensidade:.1f}")
            if self.modo_luz:
                texto += " | MODO LUZ"
        return texto


# -------------------------------------------------------------------- desenho
# coeficientes do modelo de Blinn-Phong
KA, KD, KS, BRILHO = 0.25, 0.75, 0.6, 40.0
COMPONENTES = ["só ambiente", "ambiente + difusa", "ambiente + difusa + especular"]
programas = {}              # criados em inicializar_gl(), quando já existe contexto


def desenhar_arvore(estado):
    modelview = estado.matriz_visao() @ estado.matriz_modelo()
    if estado.normais is not None:
        desenhar_com_shader(estado, modelview)
        return

    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(T.para_opengl(modelview))
    vbo_v, vbo_c, _ = estado.vbos
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


def desenhar_com_shader(estado, modelview):
    """Tubos: as matrizes do grupo e a luz vão como uniforms e a iluminação
    é calculada pelos nossos shaders, sem o pipeline fixo."""
    prog = programas[estado.sombreamento]
    prog.usar()
    prog.mat4("u_projecao", estado.matriz_projecao())
    prog.mat4("u_modelview", modelview)
    prog.mat3("u_normal", shaders.matriz_normal(modelview))

    # a luz está no mundo; o shader trabalha no espaço da câmera: aplica V
    luz_camera = estado.matriz_visao() @ np.append(estado.luz, 1.0)
    prog.vec3("u_luz_pos", luz_camera[:3])
    prog.float("u_intensidade", estado.intensidade)
    prog.float("u_ka", KA)
    prog.float("u_kd", KD)
    prog.float("u_ks", KS)
    prog.float("u_brilho", BRILHO)
    prog.float("u_usar_difusa", 1.0 if estado.componentes >= 1 else 0.0)
    prog.float("u_usar_especular", 1.0 if estado.componentes >= 2 else 0.0)

    # VBOs -> atributos (location 0, 1, 2). No Flat, o atributo de normal
    # recebe as normais de FACE em vez das normais dos vértices.
    prog.float("u_destacar_tras", 1.0 if estado.destacar_tras else 0.0)

    vbo_v, vbo_c, vbo_n, vbo_nf = estado.vbos
    vbo_normal = vbo_nf if estado.sombreamento == "Flat" else vbo_n
    for local, vbo in enumerate((vbo_v, vbo_c, vbo_normal)):
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(local)
        glVertexAttribPointer(local, 3, GL_FLOAT, GL_FALSE, 0, None)

    total = len(estado.vertices) // 3
    modo = CULLING[estado.culling]
    alfa = OPACIDADES[estado.opacidade]
    prog.float("u_alfa", alfa)
    transparente = alfa < 1.0
    painter = transparente and ORDENS[estado.ordem].startswith("Painter")

    # lista de triângulos: filtrada pelo culling próprio e/ou ordenada (Painter)
    tris = None
    if modo == "próprio (N·V)":
        tris = triangulos_de_frente(estado, modelview)
    if painter:
        tris = ordenar_tras_para_frente(estado, modelview,
                                        np.arange(total) if tris is None else tris)

    if modo == "OpenGL":
        # o OpenGL decide pela ORDEM dos vértices já projetados na tela:
        # anti-horário = frente (glFrontFace), horário = trás (descartada)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        glCullFace(GL_BACK)
    if painter:
        # já ordenados: o Z-Buffer continua TESTANDO (objetos opacos tapam),
        # mas não GRAVA, para um vaso transparente não esconder o de trás
        glDepthMask(GL_FALSE)

    if tris is None:
        glDrawArrays(GL_TRIANGLES, 0, len(estado.vertices))
        estado.tri_desenhados = total     # o OpenGL não informa quantos descartou
    else:
        indices = (tris[:, None] * 3 + np.arange(3)).astype(np.uint32).ravel()
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, estado.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STREAM_DRAW)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        estado.tri_desenhados = len(tris)

    glDepthMask(GL_TRUE)
    glDisable(GL_CULL_FACE)

    for local in range(3):
        glDisableVertexAttribArray(local)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(0)


def triangulos_de_frente(estado, modelview):
    """Back-Face Culling próprio: índices dos triângulos com N . V > 0.

    No espaço da câmera, N é a normal da face e V aponta do triângulo para
    o observador. Na perspectiva o observador está na origem, então
    V = -centroide; na ortográfica todos os raios são paralelos a +z,
    então V = (0, 0, 1). Se N . V <= 0, a face está de costas e é descartada.
    """
    normais = estado.tri_normal @ shaders.matriz_normal(modelview).T
    if estado.perspectiva:
        centros = estado.tri_centro @ modelview[:3, :3].T + modelview[:3, 3]
        n_dot_v = -(normais * centros).sum(axis=1)
    else:
        n_dot_v = normais[:, 2]
    return np.nonzero(n_dot_v > 0)[0]


def ordenar_tras_para_frente(estado, modelview, tris):
    """Painter's Algorithm: ordena os triângulos do mais distante ao mais próximo.

    No espaço da câmera o observador olha para -z: quanto MAIS negativo o z
    do centroide, mais longe está o triângulo. Desenhando nessa ordem, cada
    triângulo transparente é composto sobre tudo o que está atrás dele,
    como um pintor que pinta o fundo primeiro.
    """
    z = estado.tri_centro[tris] @ modelview[2, :3] + modelview[2, 3]
    return tris[np.argsort(z, kind="stable")]


def desenhar_luz(estado):
    """Marcador amarelo na posição da luz (só a matriz de visão: a luz é do mundo)."""
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(T.para_opengl(estado.matriz_visao()))
    glPointSize(14.0 if estado.modo_luz else 9.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.8, 0.0)
    glVertex3f(*estado.luz)
    glEnd()


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


def mostrar_profundidade(estado):
    """Lê o Z-Buffer e desenha a profundidade em tons de cinza (perto = escuro).

    O buffer guarda d em [0, 1], que NÃO é linear na distância quando a
    projeção é perspectiva. A conversão para a distância z da câmera é a
    inversa da projeção:  z = 2 n f / ((f + n) - (2d - 1)(f - n)).
    """
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    d = np.frombuffer(glReadPixels(0, 0, LARGURA, ALTURA, GL_DEPTH_COMPONENT, GL_FLOAT),
                      dtype=np.float32).reshape(ALTURA, LARGURA).astype(float)
    fundo = d >= 1.0
    if estado.perspectiva:
        z = 2 * PERTO * LONGE / ((LONGE + PERTO) - (2 * d - 1) * (LONGE - PERTO))
    else:
        z = PERTO + d * (LONGE - PERTO)
    cinza = np.full_like(z, 255.0)
    if (~fundo).any():
        zmin, zmax = z[~fundo].min(), z[~fundo].max()
        cinza[~fundo] = 20 + 200 * (z[~fundo] - zmin) / max(zmax - zmin, 1e-9)
    img = np.repeat(cinza[:, :, None], 3, axis=2).astype(np.uint8)
    glClear(GL_COLOR_BUFFER_BIT)
    glWindowPos2d(0, 0)
    glDrawPixels(LARGURA, ALTURA, GL_RGB, GL_UNSIGNED_BYTE, np.ascontiguousarray(img))


def desenhar_quadro(estado, fonte):
    # o Z-Buffer é lido da janela; o FBO multiamostrado não pode ser lido direto
    usar_msaa = estado.msaa and estado.fbo_msaa is not None and not estado.ver_profundidade
    if usar_msaa:                              # desenha a cena no FBO multiamostrado
        glBindFramebuffer(GL_FRAMEBUFFER, estado.fbo_msaa)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Z-Buffer: cada fragmento só é escrito se estiver mais perto que o já gravado
    if estado.zbuffer:
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
    else:
        glDisable(GL_DEPTH_TEST)               # vale o último desenhado (ordem do VBO)
    glMatrixMode(GL_PROJECTION)               # projeção vale para eixos e árvore
    glLoadMatrixf(T.para_opengl(estado.matriz_projecao()))

    if estado.eixos:
        desenhar_eixos(estado)
    desenhar_arvore(estado)
    if estado.normais is not None:
        desenhar_luz(estado)

    if usar_msaa:                              # resolve: média das amostras -> janela
        glBindFramebuffer(GL_READ_FRAMEBUFFER, estado.fbo_msaa)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, LARGURA, ALTURA, 0, 0, LARGURA, ALTURA,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    if estado.ver_profundidade:
        mostrar_profundidade(estado)

    glDisable(GL_DEPTH_TEST)                  # sobreposições 2D sempre por cima
    if estado.modo != 2:
        desenhar_barra_cores(estado, fonte)
    for i, linha in enumerate(reversed(estado.descricao().split("\n"))):
        desenhar_texto(fonte, linha, 10, 10 + 22 * i)


# -------------------------------------------------------------------- entrada
def tratar_tecla_luz(estado, tecla):
    """No modo luz, setas e PageUp/PageDown movem a fonte e + / - mudam a intensidade."""
    passo = 0.2
    movimentos = {pygame.K_LEFT: (-passo, 0, 0), pygame.K_RIGHT: (passo, 0, 0),
                  pygame.K_UP: (0, passo, 0), pygame.K_DOWN: (0, -passo, 0),
                  pygame.K_PAGEUP: (0, 0, passo), pygame.K_PAGEDOWN: (0, 0, -passo)}
    if tecla in movimentos:
        estado.luz = estado.luz + np.array(movimentos[tecla])
        return True
    if tecla in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        estado.intensidade = min(estado.intensidade + 0.1, 2.0)
        return True
    if tecla in (pygame.K_MINUS, pygame.K_KP_MINUS):
        estado.intensidade = max(estado.intensidade - 0.1, 0.0)
        return True
    return False


def tratar_tecla(estado, tecla):
    passo = 0.05
    if estado.modo_luz and tratar_tecla_luz(estado, tecla):
        return
    if tecla == pygame.K_g:
        nomes = list(shaders.FONTES)
        estado.sombreamento = nomes[(nomes.index(estado.sombreamento) + 1) % len(nomes)]
    elif tecla == pygame.K_k:
        estado.componentes = (estado.componentes + 1) % len(COMPONENTES)
    elif tecla == pygame.K_l:
        estado.modo_luz = not estado.modo_luz
    elif tecla == pygame.K_b:
        estado.culling = (estado.culling + 1) % len(CULLING)
    elif tecla == pygame.K_d:
        estado.zbuffer = not estado.zbuffer
    elif tecla == pygame.K_v:
        estado.ver_profundidade = not estado.ver_profundidade
    elif tecla == pygame.K_f:
        estado.destacar_tras = not estado.destacar_tras
    elif tecla == pygame.K_o:
        estado.opacidade = (estado.opacidade + 1) % len(OPACIDADES)
    elif tecla == pygame.K_n:
        estado.ordem = (estado.ordem + 1) % len(ORDENS)
    elif tecla == pygame.K_e:
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
    programas.update(shaders.criar_programas())
    print("OpenGL:", glGetString(GL_VERSION).decode(),
          "| GLSL:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())


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
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree3D_Nterm064.vtk"
    arvore = ler_vtk(caminho)

    pygame.display.init()             # só vídeo e fonte (evita iniciar o áudio)
    pygame.font.init()
    pygame.display.set_mode((LARGURA, ALTURA), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption(f"TP1 CCO - {arvore.nome}")
    fonte = pygame.font.Font(None, 24)

    inicializar_gl()

    estado = Estado(arvore)
    estado.modo = 1 if arvore.eh_3d else 0          # TP2: modelos 3D abrem em tubos
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
