# TP1 · Visualização Geométrica, Rasterização e Recorte de Árvores Arteriais CCO

**BCC327 – Computação Gráfica · UFOP · 2026.2**
Prof. Dr. Rafael Alves Bonfim de Queiroz

## Integrantes

| Nome | Matrícula |
|---|---|
| Brenda Gabrielle Alves Nascimento | 24.1.4011 |

## Visão geral

A aplicação é formada por dois programas que compartilham os mesmos módulos:

- **Visualizador CCO** (`src/visualizador.py`): carrega os modelos 2D e 3D, desenha a árvore como linhas, faixas ou tubos coloridos pelo raio, aplica translação, rotação e escala, alterna vistas do observador e projeções ortográfica e perspectiva, e liga/desliga o MSAA.
- **Modo didático** (`src/modo_didatico.py`): um framebuffer de 80 x 60 pixels ampliados em que **apenas os algoritmos do grupo escrevem**. Demonstra Bresenham/DDA, Scanline Fill, supersampling, Cohen–Sutherland, Liang–Barsky, Sutherland–Hodgman e Weiler–Atherton.

## Tecnologia e dependências

| Item | Versão |
|---|---|
| Python | 3.10 ou superior |
| pygame-ce | 2.5.2 |
| PyOpenGL | 3.1.7 |
| NumPy | 2.1.3 |
| OpenGL | 2.1 ou superior (perfil de compatibilidade); testado com Mesa 23 (llvmpipe, OpenGL 4.5) |

As versões estão fixadas em `requirements.txt`.

## Instalação

### Linux / WSL (Ubuntu)

```bash
sudo apt install -y python3-venv libgl1 libopengl0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Arquivos VTK

Os 12 modelos fornecidos pelo professor já estão em `data/`:

```
data/tree2D_Nterm008.vtk ... data/tree2D_Nterm256.vtk   (modelos 2D)
data/tree3D_Nterm016.vtk ... data/tree3D_Nterm512.vtk   (modelos 3D)
```

Os comandos abaixo devem ser executados a partir da **raiz do repositório**, para que os caminhos `data/...` e `report/figures/` funcionem.

## Execução

```bash
python src/visualizador.py                              # padrão: tree2D_Nterm064
python src/visualizador.py data/tree3D_Nterm256.vtk     # qualquer modelo 2D ou 3D

python src/modo_didatico.py                             # padrão: tree2D_Nterm016
python src/modo_didatico.py data/tree2D_Nterm032.vtk    # qualquer modelo 2D
```

Programas de linha de comando (exemplos numéricos usados no relatório):

```bash
python src/leitor_vtk.py data/*.vtk          # pontos, segmentos e raio mín./máx. de cada modelo
python src/transformacoes.py                 # testes das matrizes (ordem das transformações, projeções)
python src/rasterizacao.py                   # tabela passo a passo do Bresenham (20,10)-(30,18)
python src/rasterizacao.py 0 0 7 3           # Bresenham de um segmento qualquer
python src/rasterizacao.py scanline          # tabela linha a linha do Scanline Fill
python src/recorte.py                        # rastro de Cohen–Sutherland e Liang–Barsky + comparação
python src/recorte.py poligono               # rastro de Sutherland–Hodgman e Weiler–Atherton
```

Todas as figuras do relatório e da apresentação podem ser regeradas com um único comando (uma janela OpenGL abre por alguns segundos):

```bash
python src/gerar_figuras.py                  # 36 figuras em report/figures/
```

## Comandos do visualizador

| Entrada | Ação |
|---|---|
| Arrastar (botão esquerdo) | Rotação em torno de x e y |
| Roda do mouse | Zoom (aproxima/afasta a câmera) |
| `Z` / `X` | Rotação em torno de z |
| Setas | Translação em x e y |
| `PageUp` / `PageDown` | Translação em z |
| `,` / `.` | Escala (diminui/aumenta) |
| `1` `2` `3` `4` | Vista frontal / lateral / superior / isométrica |
| `P` | Projeção ortográfica / perspectiva |
| `A` | Mostra/esconde os eixos do mundo |
| `R` | Restaura transformações e vista |
| `E` | Representação: linhas + cor / espessura + cor / espessura sólida |
| `C` | Mapa de cores: jet / viridis |
| `+` / `-` | Exagero da espessura |
| `M` | Liga/desliga o MSAA 4x |
| `S` | Salva a tela atual em `report/figures/` |
| `ESC` | Sai |

## Comandos do modo didático

| Tecla | Ação |
|---|---|
| `1` | Árvore rasterizada (Bresenham ou DDA) |
| `2` | Segmento livre: clique em dois pixels; a tabela do Bresenham sai no terminal |
| `3` | Polígono preenchido com Scanline Fill |
| `4` | Antialiasing: mesma cena sem AA (esquerda) e com SSAA (direita) |
| `5` | Recorte de segmentos da árvore |
| `6` | Recorte de polígonos |
| `B` | (1, 2) Algoritmo de segmentos: Bresenham / DDA |
| `L` | (1, 2) Mostra/esconde a reta ideal |
| `N`, `Enter` | (3) Novo polígono: clique nos vértices e feche com Enter |
| `Espaço` | (3) Anima o preenchimento linha a linha |
| `K` | (4) Fator de supersampling: 2 / 4 / 8 |
| `T` | (4) Cena: leque de linhas / árvore CCO |
| Arrastar | (5, 6) Desenha uma nova janela de recorte |
| `G` | (5) Cohen–Sutherland / Liang–Barsky |
| `W` | (6) Sutherland–Hodgman / Weiler–Atherton |
| `P` | (6) Sujeito: região de interesse / forma em U |
| `J` | (6) Janela: retângulo / hexágono / L côncavo |
| `V` | (5, 6) Vista: sobreposto / antes / depois |
| `I` | (5, 6) Imprime no terminal a tabela do recorte |
| `S` | Salva a tela atual em `report/figures/` |
| `ESC` | Sai |

## Roteiro mínimo de teste

1. **Leitura:** `python src/leitor_vtk.py data/*.vtk`. Cada modelo deve ter `pontos = 2 x Nterm` e `segmentos = pontos - 1`.
2. **Primitivas e cores:** `python src/visualizador.py`. Aperte `E` duas vezes (linhas, faixas, sólido) e `C` (jet, viridis).
3. **Transformações e observador:** `python src/visualizador.py data/tree3D_Nterm256.vtk`. Arraste com o mouse, use setas, `Z`, `,` e `.`; os eixos do mundo (`A`) ficam parados. Teste as vistas `1` a `4` e alterne `P` com a câmera próxima.
4. **Tubos e MSAA:** no mesmo modelo, `E` para tubos, `4` para a vista isométrica, aproxime e alterne `M`.
5. **Bresenham:** `python src/modo_didatico.py`, tecla `1`, depois `2` e clique em dois pixels.
6. **Scanline:** tecla `3`, depois `Espaço`.
7. **Antialiasing:** tecla `4`, depois `K` e `T`.
8. **Recorte de segmentos:** tecla `5`, arraste uma janela, alterne `V` e `G`.
9. **Recorte de polígonos:** tecla `6`, `P` (forma em U), `V` até "depois" e alterne `W`; depois `J` até "L côncavo" e alterne `W`.

## O que foi implementado pelo grupo e o que vem de bibliotecas

### Implementado pelo grupo

| Funcionalidade | Arquivo |
|---|---|
| Leitor de VTK legado ASCII (POINTS, LINES, CELL_DATA) | `leitor_vtk.py` |
| Mapas de cores contínuos jet e viridis (interpolação linear) | `cores.py` |
| Geometria das faixas 2D (retângulos + discos) e dos tubos 3D (cilindros + esferas, com normais) | `geometria.py` |
| Matrizes de translação, rotação, escala, câmera (look-at), projeção ortográfica e perspectiva | `transformacoes.py` |
| Rasterização de segmentos: Bresenham (8 octantes) e DDA | `rasterizacao.py` |
| Preenchimento de polígonos: Scanline Fill com tabela de arestas e arestas ativas | `rasterizacao.py` |
| Supersampling (SSAA) com filtro caixa | `antialiasing.py` |
| Recorte de segmentos: Cohen–Sutherland e Liang–Barsky | `recorte.py` |
| Recorte de polígonos: Sutherland–Hodgman (janela convexa) e Weiler–Atherton (diferencial) | `recorte.py` |
| Framebuffer do modo didático e todas as interações | `modo_didatico.py` |

### Realizado com auxílio de bibliotecas

| Funcionalidade | Biblioteca |
|---|---|
| Janela, teclado, mouse e texto na tela | pygame-ce |
| Rasterização final dos triângulos e linhas no visualizador, teste de profundidade | OpenGL (via PyOpenGL), alimentado com as matrizes e a geometria do grupo |
| Iluminação dos tubos (luz direcional + ambiente, pipeline fixo) | OpenGL |
| MSAA (framebuffer multiamostrado e `glBlitFramebuffer`) | OpenGL |
| Envio dos vértices para a GPU (VBOs) | OpenGL |
| Operações vetoriais e de matrizes | NumPy |
| Ampliação do framebuffer didático e linha antisserrilhada da "reta ideal" (apenas referência visual) | pygame-ce |

Nenhum pixel do modo didático é gerado pela biblioteca: o pygame-ce apenas amplia a matriz NumPy preenchida pelos algoritmos do grupo.

## Observações

- No WSL, se o visualizador ficar lento com os modelos maiores em tubos, é possível tentar usar a GPU com `GALLIUM_DRIVER=d3d12 python src/visualizador.py ...`.
- O raio nos arquivos VTK está em mm e as coordenadas em m; a espessura desenhada é `raio x 0,001 x exagero` (exagero padrão 2x).

## Estrutura

```
.
├── src/
│   ├── leitor_vtk.py
│   ├── cores.py
│   ├── geometria.py
│   ├── transformacoes.py
│   ├── rasterizacao.py
│   ├── antialiasing.py
│   ├── recorte.py
│   ├── visualizador.py
│   ├── modo_didatico.py
│   └── gerar_figuras.py
├── data/            modelos VTK
├── report/          relatório (LaTeX) e figuras
├── presentation/    apresentação (Beamer) e figuras
├── demo/            GIF de demonstração (opcional)
├── requirements.txt
└── README.md
```
