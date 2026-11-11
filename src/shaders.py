"""
shaders.py

Programas de shader (GLSL) escritos pelo grupo e utilitários para compilá-los.

O pipeline programável substitui a iluminação fixa do OpenGL: as equações
de iluminação passam a ser código nosso, executado na GPU.
    vertex shader:   roda uma vez por VÉRTICE
    fragment shader: roda uma vez por PIXEL coberto pelo triângulo

Os três modelos de sombreamento usam a MESMA equação de iluminação
(Blinn-Phong); o que muda é ONDE ela é avaliada:
    Flat:    uma vez por triângulo, com a normal da face
    Gouraud: uma vez por vértice; as cores são interpoladas
    Phong:   uma vez por pixel; as NORMAIS são interpoladas

Convenção de atributos (layout location):
    0 = posição (x, y, z)    1 = cor (r, g, b)    2 = normal (nx, ny, nz)
"""

import numpy as np
from OpenGL.GL import *

# ---------------------------------------------------------------- iluminação
# Modelo de Blinn-Phong com uma luz pontual, tudo no espaço da câmera
# (a câmera está na origem, então a direção para o observador é -pos).
#   I = ka*cor + intensidade * (kd*cor*max(N.L, 0) + ks*max(N.H, 0)^brilho)
#   L = direção do ponto para a luz,  V = direção do ponto para a câmera
#   H = (L + V) normalizado: vetor intermediário (half vector)
ILUMINACAO = """
uniform vec3  u_luz_pos;        // posição da luz no espaço da câmera
uniform float u_intensidade;    // intensidade da fonte
uniform float u_ka;             // coeficiente ambiente
uniform float u_kd;             // coeficiente difuso
uniform float u_ks;             // coeficiente especular
uniform float u_brilho;         // expoente especular (quanto maior, menor o brilho)
uniform float u_usar_difusa;    // 1.0 liga, 0.0 desliga (para demonstração)
uniform float u_usar_especular;

vec3 iluminar(vec3 N, vec3 pos, vec3 cor) {
    N = normalize(N);
    vec3 L = normalize(u_luz_pos - pos);
    vec3 V = normalize(-pos);
    vec3 H = normalize(L + V);

    vec3 ambiente  = u_ka * cor;
    vec3 difusa    = u_kd * cor * max(dot(N, L), 0.0);
    float lado     = dot(N, L) > 0.0 ? 1.0 : 0.0;     // sem brilho no lado escuro
    vec3 especular = u_ks * vec3(1.0) * pow(max(dot(N, H), 0.0), u_brilho) * lado;

    return ambiente + u_intensidade * (u_usar_difusa * difusa + u_usar_especular * especular);
}
"""

CABECALHO_VERT = """
#version 330 core
layout(location = 0) in vec3 a_posicao;
layout(location = 1) in vec3 a_cor;
layout(location = 2) in vec3 a_normal;
uniform mat4 u_projecao;     // P
uniform mat4 u_modelview;    // V . M
uniform mat3 u_normal;       // transposta da inversa de (V . M) 3x3
"""

# ---------------------------------------------------------------- Flat
# A normal recebida é a da FACE (igual nos 3 vértices). O qualificador
# 'flat' desliga a interpolação: o triângulo inteiro usa a cor de um vértice.
FLAT_VERT = CABECALHO_VERT + ILUMINACAO + """
flat out vec3 v_cor;
void main() {
    vec4 pos = u_modelview * vec4(a_posicao, 1.0);
    gl_Position = u_projecao * pos;
    v_cor = iluminar(u_normal * a_normal, pos.xyz, a_cor);
}
"""
# Destaque de faces de trás: gl_FrontFacing é decidido pelo rasterizador a
# partir da ORDEM dos vértices na tela (anti-horário = frente).
#
# Transparência: a cor sai com alfa = u_alfa e o OpenGL compõe com o que já
# está no framebuffer (alpha blending):  C = alfa*C_novo + (1 - alfa)*C_antigo
DESTAQUE = """
uniform float u_destacar_tras;
uniform float u_alfa;            // opacidade: 1 = opaco, 0 = invisível
vec4 destacar(vec4 cor) {
    vec4 saida = (u_destacar_tras > 0.5 && !gl_FrontFacing) ? vec4(0.9, 0.1, 0.1, 1.0) : cor;
    return vec4(saida.rgb, u_alfa);
}
"""

FLAT_FRAG = """
#version 330 core
flat in vec3 v_cor;
out vec4 f_cor;
""" + DESTAQUE + """
void main() { f_cor = destacar(vec4(v_cor, 1.0)); }
"""

# ---------------------------------------------------------------- Gouraud
# Iluminação por VÉRTICE; o rasterizador interpola as CORES.
GOURAUD_VERT = CABECALHO_VERT + ILUMINACAO + """
out vec3 v_cor;
void main() {
    vec4 pos = u_modelview * vec4(a_posicao, 1.0);
    gl_Position = u_projecao * pos;
    v_cor = iluminar(u_normal * a_normal, pos.xyz, a_cor);
}
"""
GOURAUD_FRAG = """
#version 330 core
in vec3 v_cor;
out vec4 f_cor;
""" + DESTAQUE + """
void main() { f_cor = destacar(vec4(v_cor, 1.0)); }
"""

# ---------------------------------------------------------------- Phong
# O vertex shader só repassa posição e normal; o rasterizador interpola
# as NORMAIS e a iluminação é feita por PIXEL no fragment shader.
PHONG_VERT = CABECALHO_VERT + """
out vec3 v_pos;
out vec3 v_normal;
out vec3 v_cor;
void main() {
    vec4 pos = u_modelview * vec4(a_posicao, 1.0);
    gl_Position = u_projecao * pos;
    v_pos = pos.xyz;
    v_normal = u_normal * a_normal;
    v_cor = a_cor;
}
"""
PHONG_FRAG = """
#version 330 core
in vec3 v_pos;
in vec3 v_normal;            // interpolada: precisa ser renormalizada
in vec3 v_cor;
out vec4 f_cor;
""" + ILUMINACAO + DESTAQUE + """
void main() { f_cor = destacar(vec4(iluminar(v_normal, v_pos, v_cor), 1.0)); }
"""

FONTES = {
    "Flat": (FLAT_VERT, FLAT_FRAG),
    "Gouraud": (GOURAUD_VERT, GOURAUD_FRAG),
    "Phong": (PHONG_VERT, PHONG_FRAG),
}

# ---------------------------------------------------------------- Textura
# Plano de referência: o vertex shader repassa (u, v); o rasterizador as
# interpola; o fragment shader busca a cor da textura nesse ponto e a ilumina.
TEXTURA_VERT = """
#version 330 core
layout(location = 0) in vec3 a_posicao;
layout(location = 1) in vec2 a_uv;          // coordenadas de textura do vértice
uniform mat4 u_projecao;
uniform mat4 u_modelview;
uniform mat3 u_normal;
out vec3 v_pos;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    vec4 pos = u_modelview * vec4(a_posicao, 1.0);
    gl_Position = u_projecao * pos;
    v_pos = pos.xyz;
    v_normal = u_normal * vec3(0.0, 1.0, 0.0);   // o plano é horizontal
    v_uv = a_uv;
}
"""

TEXTURA_FRAG = """
#version 330 core
in vec3 v_pos;
in vec3 v_normal;
in vec2 v_uv;                  // interpolada para este pixel
out vec4 f_cor;
uniform sampler2D u_textura;
uniform int u_modo;            // 0 NEAREST, 1 bilinear OpenGL, 2 bilinear própria, 3 (u, v)
""" + ILUMINACAO + """
// Filtragem bilinear feita por nós: lê os 4 texels vizinhos com texelFetch
// (que NÃO filtra) e interpola. Centros dos texels em (i + 0,5) / n.
vec3 bilinear_propria(vec2 uv) {
    ivec2 n = textureSize(u_textura, 0);
    vec2 t = uv * vec2(n) - 0.5;
    ivec2 i0 = ivec2(floor(t));
    vec2 f = t - vec2(i0);
    ivec2 lim = n - 1;
    vec3 c00 = texelFetch(u_textura, clamp(i0,               ivec2(0), lim), 0).rgb;
    vec3 c10 = texelFetch(u_textura, clamp(i0 + ivec2(1, 0), ivec2(0), lim), 0).rgb;
    vec3 c01 = texelFetch(u_textura, clamp(i0 + ivec2(0, 1), ivec2(0), lim), 0).rgb;
    vec3 c11 = texelFetch(u_textura, clamp(i0 + ivec2(1, 1), ivec2(0), lim), 0).rgb;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    if (u_modo == 3) {                       // mostra as próprias coordenadas
        f_cor = vec4(v_uv.x, v_uv.y, 0.0, 1.0);
        return;
    }
    vec3 cor = (u_modo == 2) ? bilinear_propria(v_uv) : texture(u_textura, v_uv).rgb;
    f_cor = vec4(iluminar(v_normal, v_pos, cor), 1.0);
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

    def int(self, nome, x):
        glUniform1i(self._local(nome), int(x))


def criar_programas():
    """Compila os três modelos de sombreamento (chamar com contexto OpenGL ativo)."""
    programas = {nome: Programa(v, f) for nome, (v, f) in FONTES.items()}
    programas["Textura"] = Programa(TEXTURA_VERT, TEXTURA_FRAG)
    return programas


def matriz_normal(modelview):
    """Matriz que leva normais para o espaço da câmera: (M^-1)^T do bloco 3x3.

    Com escala não uniforme, transformar a normal pela própria modelview a
    deixaria inclinada em relação à superfície; a transposta da inversa
    preserva a perpendicularidade.
    """
    return np.linalg.inv(modelview[:3, :3]).T
