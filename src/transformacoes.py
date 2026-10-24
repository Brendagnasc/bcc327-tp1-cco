"""
transformacoes.py

Matrizes 4x4 de transformação geométrica e de câmera, implementadas pelo
grupo com NumPy (coordenadas homogêneas, convenção de vetor coluna: p' = M p).

O OpenGL guarda matrizes por coluna (column-major), então antes de enviar
uma matriz com glLoadMatrixf ela é transposta em para_opengl().

Pipeline usado no visualizador:
    p_tela ~ P . V . M . p_modelo
    M: modelo -> mundo   (translação, rotação, escala)
    V: mundo  -> câmera  (olhar_para)
    P: câmera -> recorte (ortografica ou perspectiva)
"""

import numpy as np


def translacao(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]], dtype=float)


def escala(sx, sy, sz):
    return np.array([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0, 1]], dtype=float)


def rotacao_x(graus):
    c, s = np.cos(np.radians(graus)), np.sin(np.radians(graus))
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rotacao_y(graus):
    c, s = np.cos(np.radians(graus)), np.sin(np.radians(graus))
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rotacao_z(graus):
    c, s = np.cos(np.radians(graus)), np.sin(np.radians(graus))
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def olhar_para(olho, alvo, cima):
    """Matriz de visualização (mundo -> câmera).

    Monta a base ortonormal da câmera:
        w: aponta do alvo para o olho (a câmera olha para -w)
        u: direita da câmera  = cima x w
        v: cima da câmera     = w x u
    e então expressa cada ponto do mundo nessa base, com o olho na origem.
    """
    olho, alvo, cima = (np.asarray(x, dtype=float) for x in (olho, alvo, cima))
    w = olho - alvo
    w /= np.linalg.norm(w)
    u = np.cross(cima, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    rot = np.identity(4)
    rot[0, :3], rot[1, :3], rot[2, :3] = u, v, w     # linhas = eixos da câmera
    return rot @ translacao(*(-olho))                # leva o olho para a origem, depois gira


def ortografica(esq, dir, baixo, cima, perto, longe):
    """Mapeia a caixa [esq,dir]x[baixo,cima]x[-perto,-longe] no cubo [-1,1]^3.

    Não há divisão por profundidade: retas paralelas continuam paralelas
    e o tamanho aparente não depende da distância.
    """
    return np.array([
        [2 / (dir - esq), 0, 0, -(dir + esq) / (dir - esq)],
        [0, 2 / (cima - baixo), 0, -(cima + baixo) / (cima - baixo)],
        [0, 0, -2 / (longe - perto), -(longe + perto) / (longe - perto)],
        [0, 0, 0, 1]], dtype=float)


def perspectiva(fovy_graus, aspecto, perto, longe):
    """Tronco de pirâmide (frustum) simétrico mapeado no cubo [-1,1]^3.

    A última linha copia -z para w; na divisão por w (feita pelo hardware)
    objetos mais distantes ficam menores.
    """
    f = 1.0 / np.tan(np.radians(fovy_graus) / 2)
    return np.array([
        [f / aspecto, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (longe + perto) / (perto - longe), 2 * longe * perto / (perto - longe)],
        [0, 0, -1, 0]], dtype=float)


def aplicar(M, pontos):
    """Aplica M a um array (N, 3) de pontos, com divisão por w."""
    homog = np.hstack([pontos, np.ones((len(pontos), 1))]) @ M.T
    return homog[:, :3] / homog[:, 3:4]


def para_opengl(M):
    """Converte para o formato column-major esperado pelo glLoadMatrixf."""
    return np.ascontiguousarray(M.T, dtype=np.float32)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    p = np.array([[1.0, 0.0, 0.0]])

    print("rotacao_z(90) de (1,0,0):", aplicar(rotacao_z(90), p)[0])
    print("translacao(2,3,4) de (1,0,0):", aplicar(translacao(2, 3, 4), p)[0])
    print("escala(2,2,2) de (1,0,0):", aplicar(escala(2, 2, 2), p)[0])

    # Ordem importa: girar e depois transladar != transladar e depois girar
    print("T(2,0,0) . Rz(90) de (1,0,0):", aplicar(translacao(2, 0, 0) @ rotacao_z(90), p)[0])
    print("Rz(90) . T(2,0,0) de (1,0,0):", aplicar(rotacao_z(90) @ translacao(2, 0, 0), p)[0])

    V = olhar_para([0, 0, 5], [0, 0, 0], [0, 1, 0])
    print("origem do mundo vista da câmera em (0,0,5):", aplicar(V, np.zeros((1, 3)))[0])

    P = perspectiva(45, 1.0, 0.1, 100)
    perto = aplicar(P @ V, np.array([[1.0, 0, 0]]))[0]
    longe = aplicar(P @ V, np.array([[1.0, 0, -5]]))[0]
    print("perspectiva: x de (1,0,0) =", perto[0], " x de (1,0,-5) =", longe[0])
