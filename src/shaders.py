"""
shaders.py

Programas de shader (GLSL) escritos pelo grupo e utilitários para compilá-los.

O pipeline programável substitui a iluminação fixa do OpenGL: as equações
de iluminação passam a ser código nosso, executado na GPU.
    vertex shader:   roda uma vez por VÉRTICE (posição na tela e, no
                     Gouraud, também a cor iluminada)
    fragment shader: roda uma vez por PIXEL coberto pelo triângulo

Convenção de atributos (layout location):
    0 = posição (x, y, z)    1 = cor (r, g, b)    2 = normal (nx, ny, nz)
"""

import numpy as np
from OpenGL.GL import *

# ---------------------------------------------------------------- Gouraud
# A iluminação é calculada no vertex shader (por vértice); o rasterizador
# interpola as CORES resultantes ao longo do triângulo.
GOURAUD_VERT = """
#version 330 core
layout(location = 0) in vec3 a_posicao;
layout(location = 1) in vec3 a_cor;
layout(location = 2) in vec3 a_normal;

uniform mat4 u_projecao;     // P
uniform mat4 u_modelview;    // V . M
uniform mat3 u_normal;       // transposta da inversa de (V . M) 3x3
uniform vec3 u_luz_dir;      // direção PARA a luz, no espaço da câmera
uniform float u_ka;          // coeficiente ambiente
uniform float u_kd;          // coeficiente difuso

out vec3 v_cor;

void main() {
    vec4 pos_camera = u_modelview * vec4(a_posicao, 1.0);
    gl_Position = u_projecao * pos_camera;

    vec3 N = normalize(u_normal * a_normal);
    vec3 L = normalize(u_luz_dir);
    float difusa = max(dot(N, L), 0.0);      // Lambert: cosseno entre N e L

    v_cor = a_cor * (u_ka + u_kd * difusa);
}
"""

GOURAUD_FRAG = """
#version 330 core
in vec3 v_cor;               // já interpolada entre os 3 vértices
out vec4 f_cor;

void main() {
    f_cor = vec4(v_cor, 1.0);
}
"""


def _compilar(fonte, tipo):
    shader = glCreateShader(tipo)
    glShaderSource(shader, fonte)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        erro = glGetShaderInfoLog(shader).decode()
        nome = "vertex" if tipo == GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"erro ao compilar o {nome} shader:\n{erro}")
    return shader


class Programa:
    """Um par vertex + fragment shader ligado (linkado) num programa."""

    def __init__(self, vert, frag):
        vs = _compilar(vert, GL_VERTEX_SHADER)
        fs = _compilar(frag, GL_FRAGMENT_SHADER)
        self.id = glCreateProgram()
        glAttachShader(self.id, vs)
        glAttachShader(self.id, fs)
        glLinkProgram(self.id)
        if not glGetProgramiv(self.id, GL_LINK_STATUS):
            raise RuntimeError("erro ao ligar o programa:\n"
                               + glGetProgramInfoLog(self.id).decode())
        glDeleteShader(vs)                 # já fazem parte do programa
        glDeleteShader(fs)
        self._locais = {}

    def usar(self):
        glUseProgram(self.id)

    def _local(self, nome):
        if nome not in self._locais:
            self._locais[nome] = glGetUniformLocation(self.id, nome)
        return self._locais[nome]

    # As matrizes do grupo são NumPy por LINHA; GL_TRUE pede ao OpenGL
    # que transponha ao enviar, então não precisamos de para_opengl().
    def mat4(self, nome, m):
        glUniformMatrix4fv(self._local(nome), 1, GL_TRUE, np.asarray(m, dtype=np.float32))

    def mat3(self, nome, m):
        glUniformMatrix3fv(self._local(nome), 1, GL_TRUE, np.asarray(m, dtype=np.float32))

    def vec3(self, nome, v):
        glUniform3f(self._local(nome), *map(float, v))

    def float(self, nome, x):
        glUniform1f(self._local(nome), float(x))


def matriz_normal(modelview):
    """Matriz que leva normais para o espaço da câmera: (M^-1)^T do bloco 3x3.

    Com escala não uniforme, transformar a normal pela própria modelview a
    deixaria inclinada em relação à superfície; a transposta da inversa
    preserva a perpendicularidade.
    """
    return np.linalg.inv(modelview[:3, :3]).T
