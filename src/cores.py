"""
cores.py

Mapas contínuos de cores implementados pelo grupo: um valor escalar (o raio)
é normalizado para t em [0, 1] e convertido em RGB por interpolação linear
entre cores de controle, canal a canal.

- "jet": arco-íris azul -> ciano -> amarelo -> vermelho, o mesmo estilo da
  figura de referência do professor. Tem alto contraste, mas não é
  perceptualmente uniforme (o amarelo e o ciano "saltam" mais que o resto).
- "viridis": aproximação do mapa perceptualmente uniforme do matplotlib;
  passos iguais em t parecem passos iguais de cor, e funciona em escala de
  cinza e para daltônicos.
"""

import numpy as np

# Cada mapa é uma lista de (t, (r, g, b)) com t crescente de 0 a 1
MAPAS = {
    "jet": [
        (0.000, (0.0, 0.0, 0.5)),
        (0.125, (0.0, 0.0, 1.0)),
        (0.375, (0.0, 1.0, 1.0)),
        (0.625, (1.0, 1.0, 0.0)),
        (0.875, (1.0, 0.0, 0.0)),
        (1.000, (0.5, 0.0, 0.0)),
    ],
    "viridis": [
        (0.00, (0.267, 0.005, 0.329)),
        (0.25, (0.229, 0.322, 0.546)),
        (0.50, (0.128, 0.567, 0.551)),
        (0.75, (0.369, 0.789, 0.383)),
        (1.00, (0.993, 0.906, 0.144)),
    ],
}


def mapear_cores(valores, vmin, vmax, nome="jet") -> np.ndarray:
    """Converte um array de escalares em um array (N, 3) de cores RGB em [0, 1]."""
    controle = MAPAS[nome]
    ts = np.array([c[0] for c in controle])
    rgbs = np.array([c[1] for c in controle])

    # Normalização: vmin vira 0 e vmax vira 1
    t = (np.asarray(valores, dtype=float) - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)

    # Interpolação linear separada para R, G e B
    return np.stack([np.interp(t, ts, rgbs[:, canal]) for canal in range(3)], axis=1)
