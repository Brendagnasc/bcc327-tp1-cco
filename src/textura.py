"""
textura.py

Mapeamento de textura implementado pelo grupo:
- gerar_textura(): cria a imagem da textura em NumPy (quadriculado com o
  contorno do domínio de perfusão), sem arquivo externo;
- amostrar_nearest / amostrar_bilinear: as duas filtragens em NumPy, usadas
  para explicar e validar o que o shader faz;
- PlanoTexturizado: um plano de referência sob a árvore, com coordenadas
  de textura (u, v) em cada vértice.

Coordenadas de textura: cada vértice recebe um par (u, v) em [0, 1] que diz
qual ponto da imagem fica "colado" nele (u = 0 borda esquerda, u = 1 borda
direita; v = 0 embaixo, v = 1 em cima). O rasterizador interpola (u, v) dentro
do triângulo e, para cada pixel, a cor é buscada na imagem nesse ponto.
"""

import numpy as np
from OpenGL.GL import *

TAMANHO = 32          # resolução da textura (32 x 32 texels): baixa de propósito,
                      # para a diferença entre as filtragens ficar visível
PLANO_Y = -1.15       # altura do plano (o modelo normalizado ocupa [-1, 1])
PLANO_MEIO = 1.6      # o plano vai de -1,6 a 1,6 em x e z


def gerar_textura(n=TAMANHO):
    """Quadriculado 4 x 4 em dois tons + círculo do domínio de perfusão."""
    j, i = np.mgrid[0:n, 0:n]                      # i = coluna (u), j = linha (v)
    celula = n // 4
    claro = ((i // celula + j // celula) % 2 == 0)
    img = np.where(claro[..., None], [235, 235, 240], [190, 198, 214]).astype(float)

    # contorno do domínio: o modelo normalizado cabe numa esfera de raio ~1;
    # no plano de meia largura PLANO_MEIO isso vira um círculo de raio n/2/PLANO_MEIO
    r = np.hypot(i + 0.5 - n / 2, j + 0.5 - n / 2)
    raio = n / 2 / PLANO_MEIO
    img[np.abs(r - raio) < 0.7] = [140, 29, 36]   # vermelho UFOP
    return img.astype(np.uint8)


def amostrar_nearest(img, u, v):
    """Cor do texel cujo centro está mais perto de (u, v)."""
    n_v, n_u = img.shape[:2]
    i = np.clip(np.floor(u * n_u).astype(int), 0, n_u - 1)
    j = np.clip(np.floor(v * n_v).astype(int), 0, n_v - 1)
    return img[j, i].astype(float)


def amostrar_bilinear(img, u, v):
    """Média ponderada dos 4 texels vizinhos (filtragem bilinear).

    Os centros dos texels ficam em (i + 0,5) / n. Para o ponto (u, v):
        t = u*n - 0,5   ->  i0 = piso(t),  fx = t - i0  (fração entre i0 e i0+1)
    e a cor é interpolada primeiro na horizontal, depois na vertical:
        c = (1-fy)*[(1-fx)*c00 + fx*c10] + fy*[(1-fx)*c01 + fx*c11]
    """
    n_v, n_u = img.shape[:2]
    tx, ty = u * n_u - 0.5, v * n_v - 0.5
    i0, j0 = np.floor(tx).astype(int), np.floor(ty).astype(int)
    fx, fy = (tx - i0)[..., None], (ty - j0)[..., None]
    i0c, i1c = np.clip(i0, 0, n_u - 1), np.clip(i0 + 1, 0, n_u - 1)
    j0c, j1c = np.clip(j0, 0, n_v - 1), np.clip(j0 + 1, 0, n_v - 1)
    c00, c10 = img[j0c, i0c].astype(float), img[j0c, i1c].astype(float)
    c01, c11 = img[j1c, i0c].astype(float), img[j1c, i1c].astype(float)
    return (1 - fy) * ((1 - fx) * c00 + fx * c10) + fy * ((1 - fx) * c01 + fx * c11)


class PlanoTexturizado:
    """Plano horizontal (dois triângulos) com coordenadas de textura."""

    def __init__(self):
        m, y = PLANO_MEIO, PLANO_Y
        # 4 cantos do plano e suas coordenadas (u, v)
        cantos = np.array([[-m, y, m], [m, y, m], [m, y, -m], [-m, y, -m]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        ordem = [0, 1, 2, 0, 2, 3]            # dois triângulos, anti-horário vistos de cima
        self.vertices = np.ascontiguousarray(cantos[ordem])
        self.uv = np.ascontiguousarray(uv[ordem])

        self.vbo_pos, self.vbo_uv = glGenBuffers(2)
        for vbo, dados in ((self.vbo_pos, self.vertices), (self.vbo_uv, self.uv)):
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, dados.nbytes, dados, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # envia a imagem para a GPU como textura 2D
        self.imagem = gerar_textura()
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, TAMANHO, TAMANHO, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, np.ascontiguousarray(self.imagem))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def desenhar(self, filtro_gl):
        """filtro_gl: GL_NEAREST ou GL_LINEAR (o modo próprio usa texelFetch)."""
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtro_gl)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtro_gl)
        for local, vbo, comp in ((0, self.vbo_pos, 3), (1, self.vbo_uv, 2)):
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glEnableVertexAttribArray(local)
            glVertexAttribPointer(local, comp, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        for local in (0, 1):
            glDisableVertexAttribArray(local)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)


if __name__ == "__main__":
    img = gerar_textura()
    # exemplo numérico: ponto na fronteira entre quatro células do quadriculado
    u, v = 0.245, 0.50
    tx, ty = u * TAMANHO - 0.5, v * TAMANHO - 0.5
    i0, j0 = int(np.floor(tx)), int(np.floor(ty))
    print(f"textura {TAMANHO}x{TAMANHO} | ponto (u, v) = ({u}, {v})")
    print(f"  t = ({tx:.2f}, {ty:.2f})  ->  texels i = {i0}, {i0 + 1}  j = {j0}, {j0 + 1}"
          f"  | fx = {tx - i0:.2f}, fy = {ty - j0:.2f}")
    for dj in (0, 1):
        for di in (0, 1):
            print(f"  c{di}{dj} = texel ({i0 + di}, {j0 + dj}) = {img[j0 + dj, i0 + di]}")
    print("  NEAREST :", amostrar_nearest(img, np.array([u]), np.array([v]))[0])
    print("  bilinear:", np.round(amostrar_bilinear(img, np.array([u]), np.array([v]))[0], 2))
