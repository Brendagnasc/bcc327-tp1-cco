"""
modo_didatico.py

Modo didático: uma grade de pixels "gigantes" (um framebuffer em NumPy)
onde só os algoritmos do grupo escrevem. Serve para mostrar, pixel a pixel,
o resultado da rasterização (e, nos próximos passos, do preenchimento,
do antialiasing e do recorte).

Usa apenas pygame-ce 2D: nenhum algoritmo de desenho da biblioteca é usado
para gerar os pixels; o pygame só amplia o framebuffer para a tela.

Uso:
    python src/modo_didatico.py [arquivo_2D.vtk]

Teclas e mouse:
    1        modo árvore: segmentos da árvore CCO rasterizados
    2        modo segmento: clique em dois pixels para traçar um segmento
    3        modo polígono: região de interesse preenchida com Scanline Fill
    4        modo antialiasing: mesma cena sem (esquerda) e com SSAA (direita)
    K        (modo 4) fator de supersampling: 2 / 4 / 8
    T        (modo 4) cena: leque de linhas / árvore CCO
    5        modo recorte: segmentos da árvore contra uma janela retangular
    arrastar (modo 5) desenha uma nova janela de recorte
    G        (modo 5) algoritmo: Cohen–Sutherland / Liang–Barsky
    V        (modo 5) vista: sobreposto / antes / depois
    I        (modo 5) imprime no terminal a tabela do recorte
    6        modo recorte de polígonos: Sutherland–Hodgman / Weiler–Atherton
    arrastar (modo 6) desenha uma nova janela de recorte
    W        (modo 6) algoritmo: Sutherland–Hodgman / Weiler–Atherton
    P        (modo 6) polígono sujeito: região de interesse / forma em U
    J        (modo 6) janela: retângulo / hexágono convexo / L côncavo
    V        (modo 6) vista: sobreposto / antes / depois
    I        (modo 6) imprime no terminal o polígono após cada aresta da janela
    N        (modo 3) novo polígono: clique nos vértices e feche com Enter
    Espaço   (modo 3) anima o preenchimento linha a linha
    B        alterna o algoritmo de segmentos: Bresenham / DDA
    L        mostra / esconde a reta ideal (contínua) por cima dos pixels
    S        salva a tela em report/figures/
    ESC      sai
"""

import sys
from pathlib import Path

import numpy as np
import pygame

from antialiasing import cena_estrela, renderizar
from cores import mapear_cores
from leitor_vtk import ler_vtk
from rasterizacao import (bresenham, dda, imprimir_rastro, imprimir_scanline,
                          scanline_linhas)
from recorte import (ALGORITMOS_RECORTE, bits, classificar, codigo_regiao, liang_barsky,
                     pontes_degeneradas, retangulo_para_poligono, sutherland_hodgman,
                     weiler_atherton)

COLUNAS, LINHAS = 80, 60          # tamanho do framebuffer, em pixels
CELULA = 12                       # cada pixel vira um quadrado de 12x12 na tela
PAINEL = 60                       # faixa inferior para texto
LARGURA, ALTURA = COLUNAS * CELULA, LINHAS * CELULA + PAINEL
BRANCO = (255, 255, 255)
ALGORITMOS = {"Bresenham": bresenham, "DDA": dda}
COR_REGIAO = (255, 214, 150)      # preenchimento do polígono
COR_BORDA = (200, 90, 0)          # contorno do polígono
COR_ARVORE_FUNDO = (150, 150, 150)
COR_DESCARTE = (205, 205, 205)    # partes removidas pelo recorte
COR_JANELA = (220, 0, 0)
JANELA_PADRAO = (22.0, 14.0, 56.0, 42.0)    # (xmin, ymin, xmax, ymax)
VISTAS_RECORTE = ["sobreposto", "antes", "depois"]
COR_PONTE = (200, 0, 200)         # arestas degeneradas do Sutherland–Hodgman
CORES_PEDACOS = [(255, 214, 150), (170, 215, 255), (190, 240, 170), (255, 190, 220)]
TIPOS_JANELA = ["retângulo", "hexágono", "L côncavo"]

# Polígono côncavo em U: com a janela padrão, a base fica de fora e
# o resultado correto seriam DOIS pedaços (os braços do U)
POLIGONO_U = [(24, 4), (56, 4), (56, 50), (46, 50), (46, 12), (34, 12), (34, 50), (24, 50)]

# Polígono côncavo de exemplo: uma região de interesse sobre a árvore
POLIGONO_PADRAO = [(16, 10), (42, 6), (64, 16), (52, 30), (66, 46),
                   (40, 40), (24, 52), (12, 30)]


class Framebuffer:
    """Matriz LINHAS x COLUNAS x RGB. Origem (0, 0) no canto inferior esquerdo,
    y para cima, como nos livros de Computação Gráfica."""

    def __init__(self):
        self.pixels = np.full((LINHAS, COLUNAS, 3), 255, dtype=np.uint8)

    def limpar(self):
        self.pixels[:] = 255

    def acender(self, lista, cor):
        for x, y in lista:
            if 0 <= x < COLUNAS and 0 <= y < LINHAS:
                self.pixels[y, x] = cor

    def para_superficie(self):
        # inverte as linhas (y para cima -> y para baixo da tela) e amplia
        imagem = self.pixels[::-1].swapaxes(0, 1)
        pequena = pygame.surfarray.make_surface(imagem)
        return pygame.transform.scale(pequena, (COLUNAS * CELULA, LINHAS * CELULA))


def centro_na_tela(x, y):
    """Centro do pixel (x, y) da grade em coordenadas da janela."""
    return ((x + 0.5) * CELULA, (LINHAS - 1 - y + 0.5) * CELULA)


def pixel_do_mouse(pos):
    mx, my = pos
    return int(mx // CELULA), LINHAS - 1 - int(my // CELULA)


def segmentos_da_arvore(arvore, colunas=COLUNAS, linhas=LINHAS, margem=3, arredondar=True):
    """Projeta a árvore 2D numa grade colunas x linhas: escala uniforme,
    centralização e (opcionalmente) arredondamento para pixels inteiros."""
    xy = arvore.pontos[:, :2]
    minimo = xy.min(axis=0)
    tamanho = (xy.max(axis=0) - minimo).max()
    util = min(colunas, linhas) - 1 - 2 * margem
    grade = (xy - minimo) / tamanho * util + margem
    grade[:, 0] += (colunas - 1 - 2 * margem - util) / 2     # centraliza em x
    grade[:, 1] += (linhas - 1 - 2 * margem - util) / 2      # centraliza em y
    if arredondar:
        grade = np.rint(grade).astype(int)
    return [(*grade[i], *grade[j]) for i, j in arvore.segmentos]


class App:
    def __init__(self, arvore):
        self.fb = Framebuffer()
        self.arvore = arvore
        self.segs_arvore = segmentos_da_arvore(arvore)
        self.cores_arvore = (mapear_cores(arvore.raios, arvore.raios.min(),
                                          arvore.raios.max(), "jet") * 255).astype(np.uint8)
        self.modo = 1
        self.algoritmo = "Bresenham"
        self.reta_ideal = True
        self.segmento = (20, 10, 30, 18)            # exemplo do relatório
        self.clique = None
        self.poligono = list(POLIGONO_PADRAO)
        self.editando = False                       # True enquanto recebe vértices
        self.anim = None                            # linha atual da animação
        self.fator = 4                              # supersampling do modo 4
        self.cena_aa = "leque"
        self.segs_reais = segmentos_da_arvore(arvore, arredondar=False)
        self.janela = JANELA_PADRAO
        self.alg_recorte = "Cohen–Sutherland"
        self.vista_recorte = 0
        self.arrasto = None                         # canto inicial ao arrastar
        self.sujeito = "região de interesse"
        self.tipo_janela = "retângulo"
        self.alg_poligono = "Sutherland–Hodgman"
        self.info = ""

    # ------------------------------------------------------------ rasterizar
    def rasterizar(self):
        self.fb.limpar()
        f = ALGORITMOS[self.algoritmo]
        total = 0
        if self.modo == 1:
            # finos primeiro, grossos por cima
            for idx in np.argsort(self.arvore.raios):
                px = f(*self.segs_arvore[idx])
                self.fb.acender(px, self.cores_arvore[idx])
                total += len(px)
            self.info = (f"Árvore {self.arvore.nome} | {self.algoritmo} | "
                         f"{len(self.segs_arvore)} segmentos | {total} pixels")
        elif self.modo == 3:
            self.rasterizar_poligono(f)
        elif self.modo == 4:
            self.rasterizar_antialiasing()
        elif self.modo == 5:
            self.rasterizar_recorte(f)
        elif self.modo == 6:
            self.rasterizar_recorte_poligono(f)
        else:
            x0, y0, x1, y1 = self.segmento
            px = f(x0, y0, x1, y1)
            self.fb.acender(px, (30, 60, 160))
            self.fb.acender([(x0, y0)], (0, 160, 0))
            self.fb.acender([(x1, y1)], (200, 0, 0))
            self.info = (f"Segmento ({x0},{y0}) a ({x1},{y1}) | {self.algoritmo} | "
                         f"{len(px)} pixels | clique em dois pixels")

    def rasterizar_poligono(self, f):
        """Região preenchida, árvore em cinza por cima e contorno por último."""
        def desenhar_arvore_cinza():
            for x0, y0, x1, y1 in self.segs_arvore:
                self.fb.acender(f(x0, y0, x1, y1), COR_ARVORE_FUNDO)

        if self.editando or len(self.poligono) < 3:
            desenhar_arvore_cinza()
            for a, b in zip(self.poligono, self.poligono[1:]):
                self.fb.acender(f(*a, *b), COR_BORDA)
            self.fb.acender(self.poligono, (0, 160, 0))
            self.info = (f"Novo polígono: {len(self.poligono)} vértices | "
                         f"clique para adicionar, Enter fecha")
            return

        linhas = scanline_linhas(self.poligono)
        visiveis = linhas if self.anim is None else linhas[:self.anim + 1]
        total = 0
        for y, _, spans in visiveis:
            for xi, xf in spans:
                self.fb.acender([(x, y) for x in range(xi, xf + 1)], COR_REGIAO)
                total += max(0, xf - xi + 1)

        desenhar_arvore_cinza()
        n = len(self.poligono)
        for i in range(n):
            self.fb.acender(f(*self.poligono[i], *self.poligono[(i + 1) % n]), COR_BORDA)
        self.fb.acender(self.poligono, (0, 160, 0))

        estado = "" if self.anim is None else f" | linha y = {visiveis[-1][0]}"
        self.info = (f"Scanline Fill | {n} vértices | {len(linhas)} linhas | "
                     f"{total} pixels internos{estado}")

    def rasterizar_antialiasing(self):
        """Metade esquerda: cena com k = 1 (serrilhada).
        Metade direita: mesma cena com supersampling k x k."""
        meia = COLUNAS // 2
        if self.cena_aa == "leque":
            cena = cena_estrela(meia, LINHAS)
        else:
            segs = segmentos_da_arvore(self.arvore, meia, LINHAS, margem=2, arredondar=False)
            ordem = np.argsort(self.arvore.raios)
            cena = [("linha", segs[i], tuple(int(c) for c in self.cores_arvore[i]))
                    for i in ordem]
        self.fb.pixels[:, :meia] = renderizar(cena, meia, LINHAS, 1)
        self.fb.pixels[:, meia:] = renderizar(cena, meia, LINHAS, self.fator)
        self.info = (f"Antialiasing | cena {self.cena_aa} | esquerda: sem AA | "
                     f"direita: SSAA {self.fator}x ({self.fator ** 2} amostras por pixel)")

    def recortar_todos(self):
        f = ALGORITMOS_RECORTE[self.alg_recorte]
        return [f(*seg, self.janela) for seg in self.segs_reais]

    def rasterizar_recorte(self, f):
        """O recorte é feito em coordenadas reais; só depois os pedaços que
        sobraram são rasterizados com Bresenham."""
        inteiro = lambda seg: tuple(int(round(v)) for v in seg)
        vista = VISTAS_RECORTE[self.vista_recorte]
        recortados = self.recortar_todos()
        ordem = np.argsort(self.arvore.raios)

        if vista in ("sobreposto", "antes"):
            cor_antes = COR_DESCARTE if vista == "sobreposto" else None
            for i in ordem:
                cor = cor_antes if cor_antes is not None else self.cores_arvore[i]
                self.fb.acender(f(*inteiro(self.segs_reais[i])), cor)
        if vista in ("sobreposto", "depois"):
            for i in ordem:
                if recortados[i] is not None:
                    self.fb.acender(f(*inteiro(recortados[i])), self.cores_arvore[i])

        cont = {"inteiro": 0, "recortado": 0, "rejeitado": 0}
        for seg, rec in zip(self.segs_reais, recortados):
            cont[classificar(seg, rec)] += 1
        self.info = (f"{self.alg_recorte} | vista {vista} | {cont['inteiro']} inteiros, "
                     f"{cont['recortado']} recortados, {cont['rejeitado']} rejeitados")

    def imprimir_recorte(self):
        xmin, ymin, xmax, ymax = self.janela
        print(f"\n{self.alg_recorte} | janela x [{xmin:.0f}, {xmax:.0f}]  y [{ymin:.0f}, {ymax:.0f}]")
        if self.alg_recorte == "Cohen–Sutherland":
            print(f"{'seg':>4}  {'c0':>4}  {'c1':>4}  {'c0|c1':>5}  {'c0&c1':>5}  resultado")
            for i, (seg, rec) in enumerate(zip(self.segs_reais, self.recortar_todos())):
                c0 = codigo_regiao(seg[0], seg[1], self.janela)
                c1 = codigo_regiao(seg[2], seg[3], self.janela)
                print(f"{i:>4}  {bits(c0)}  {bits(c1)}  {bits(c0 | c1):>5}  "
                      f"{bits(c0 & c1):>5}  {classificar(seg, rec)}")
        else:
            print(f"{'seg':>4}  {'t0':>6}  {'t1':>6}  resultado")
            for i, seg in enumerate(self.segs_reais):
                r = []
                rec = liang_barsky(*seg, self.janela, r)
                if rec is None:       # parou antes: mostra a borda que o eliminou
                    print(f"{i:>4}  {'-':>6}  {'-':>6}  rejeitado na borda {r[-1][0]}")
                else:
                    t0, t1 = r[-1][4], r[-1][5]
                    print(f"{i:>4}  {t0:6.3f}  {t1:6.3f}  {classificar(seg, rec)}")

    def poligono_sujeito(self):
        return self.poligono if self.sujeito == "região de interesse" else POLIGONO_U

    def poligono_janela(self):
        if self.tipo_janela == "retângulo":
            return retangulo_para_poligono(self.janela)
        xmin, ymin, xmax, ymax = self.janela
        q, cx, cy = (xmax - xmin) / 4, (xmin + xmax) / 2, (ymin + ymax) / 2
        if self.tipo_janela == "hexágono":       # convexo, inscrito no retângulo
            return [(xmin + q, ymin), (xmax - q, ymin), (xmax, cy),
                    (xmax - q, ymax), (xmin + q, ymax), (xmin, cy)]
        # "L côncavo": retângulo sem o quadrante superior direito
        return [(xmin, ymin), (xmax, ymin), (xmax, cy), (cx, cy), (cx, ymax), (xmin, ymax)]

    def preencher_e_contornar(self, poligono, f, cor_fundo, cor_borda):
        """Preenche com o nosso Scanline e contorna com o nosso Bresenham."""
        inteiro = lambda pt: (int(round(pt[0])), int(round(pt[1])))
        if cor_fundo is not None and len(poligono) >= 3:
            for y, _, spans in scanline_linhas(poligono):
                for xi, xf in spans:
                    self.fb.acender([(x, y) for x in range(xi, xf + 1)], cor_fundo)
        n = len(poligono)
        for i in range(n):
            self.fb.acender(f(*inteiro(poligono[i]), *inteiro(poligono[(i + 1) % n])), cor_borda)

    def recortar_poligono(self):
        """Lista de polígonos resultantes (o SH sempre devolve no máximo um)."""
        sujeito, janela = self.poligono_sujeito(), self.poligono_janela()
        if self.alg_poligono == "Weiler–Atherton":
            return weiler_atherton(sujeito, janela)
        res = sutherland_hodgman(sujeito, janela)
        return [res] if len(res) >= 3 else []

    def rasterizar_recorte_poligono(self, f):
        vista = VISTAS_RECORTE[self.vista_recorte]
        sujeito = self.poligono_sujeito()
        pedacos = self.recortar_poligono()
        self.pontes = [p for pedaco in pedacos for p in pontes_degeneradas(pedaco)]

        if vista == "antes":
            self.preencher_e_contornar(sujeito, f, COR_REGIAO, COR_BORDA)
        else:
            if vista == "sobreposto":
                self.preencher_e_contornar(sujeito, f, None, COR_DESCARTE)
            for i, pedaco in enumerate(pedacos):     # uma cor por polígono resultante
                self.preencher_e_contornar(pedaco, f, CORES_PEDACOS[i % len(CORES_PEDACOS)],
                                           COR_BORDA)

        avisos = ""
        if vista != "antes":
            if self.pontes:
                avisos += f" | {len(self.pontes)} ponte(s) degenerada(s) em magenta"
            if self.alg_poligono == "Sutherland–Hodgman" and self.tipo_janela == "L côncavo":
                avisos += " | janela côncava: SH falha"
        self.info = (f"{self.alg_poligono} | {self.sujeito} | janela {self.tipo_janela} | "
                     f"vista {vista} | {len(pedacos)} polígono(s){avisos}")

    def imprimir_recorte_poligono(self):
        janela = self.poligono_janela()
        print(f"\n{self.alg_poligono} | sujeito: {self.sujeito} | janela: {self.tipo_janela}")
        print(f"  entrada: {len(self.poligono_sujeito())} vértices")
        r = []
        if self.alg_poligono == "Sutherland–Hodgman":
            sutherland_hodgman(self.poligono_sujeito(), janela, r)
            for (ax, ay), (bx, by), saida in r:
                pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in saida)
                print(f"  após aresta ({ax:.1f}, {ay:.1f})->({bx:.1f}, {by:.1f}): "
                      f"{len(saida)} vértices: {pts}")
        else:
            pedacos = weiler_atherton(self.poligono_sujeito(), janela, r)
            print("  interseções ao longo do sujeito:")
            for x, y, tipo in r:
                print(f"    ({x:.1f}, {y:.1f}) {tipo}")
            for i, p in enumerate(pedacos):
                pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in p)
                print(f"  polígono {i + 1}: {pts}")

    def avancar_animacao(self):
        if self.anim is None:
            return
        self.anim += 1
        if self.anim >= len(scanline_linhas(self.poligono)):
            self.anim = None
        self.rasterizar()

    # ------------------------------------------------------------ desenhar
    def desenhar(self, tela, fonte):
        tela.fill(BRANCO)
        tela.blit(self.fb.para_superficie(), (0, 0))

        # linhas da grade
        cinza = (215, 215, 215)
        for c in range(COLUNAS + 1):
            pygame.draw.line(tela, cinza, (c * CELULA, 0), (c * CELULA, LINHAS * CELULA))
        for l in range(LINHAS + 1):
            pygame.draw.line(tela, cinza, (0, l * CELULA), (COLUNAS * CELULA, l * CELULA))

        # reta ideal por cima, só como referência visual
        if self.reta_ideal and self.modo in (1, 2):
            segs = self.segs_arvore if self.modo == 1 else [self.segmento]
            for x0, y0, x1, y1 in segs:
                pygame.draw.aaline(tela, (0, 0, 0), centro_na_tela(x0, y0),
                                   centro_na_tela(x1, y1))

        # linha de varredura atual durante a animação
        if self.modo == 3 and self.anim is not None:
            linhas = scanline_linhas(self.poligono)
            y = linhas[min(self.anim, len(linhas) - 1)][0]
            yt = (LINHAS - 1 - y + 0.5) * CELULA
            pygame.draw.line(tela, (220, 0, 0), (0, yt), (COLUNAS * CELULA, yt), 2)

        if self.modo == 4:
            meio = COLUNAS // 2 * CELULA
            pygame.draw.line(tela, (220, 0, 0), (meio, 0), (meio, LINHAS * CELULA), 3)
            for texto, x in (("sem antialiasing", 10), (f"SSAA {self.fator}x", meio + 10)):
                rot = fonte.render(texto, True, (20, 20, 20), BRANCO)
                tela.blit(rot, (x, 8))

        if self.modo == 5:
            xmin, ymin, xmax, ymax = self.janela
            canto_a = centro_na_tela(xmin, ymax)
            canto_b = centro_na_tela(xmax, ymin)
            ret = pygame.Rect(canto_a, (canto_b[0] - canto_a[0], canto_b[1] - canto_a[1]))
            pygame.draw.rect(tela, COR_JANELA, ret, 2)

        if self.modo == 6:
            pontos = [centro_na_tela(x, y) for x, y in self.poligono_janela()]
            pygame.draw.polygon(tela, COR_JANELA, pontos, 2)
            if VISTAS_RECORTE[self.vista_recorte] != "antes":
                for a, b in self.pontes:
                    pygame.draw.line(tela, COR_PONTE, centro_na_tela(*a),
                                     centro_na_tela(*b), 5)

        if self.clique is not None:
            pygame.draw.rect(tela, (0, 160, 0), (*[v - CELULA / 2 for v in
                             centro_na_tela(*self.clique)], CELULA, CELULA), 2)

        y_txt = LINHAS * CELULA + 8
        tela.blit(fonte.render(self.info, True, (20, 20, 20)), (10, y_txt))
        especifica = {
            1: "B algoritmo  L reta ideal",
            2: "clique em 2 pixels  B algoritmo  L reta ideal",
            3: "N novo  Enter fecha  Espaço anima",
            4: "K fator  T cena",
            5: "arraste para nova janela  G algoritmo  V vista  I tabela",
            6: "arraste para nova janela  W algoritmo  P polígono  J janela  V vista  I tabela",
        }[self.modo]
        ajuda = f"1-6 modos  |  {especifica}  |  S salvar  ESC sair"
        tela.blit(fonte.render(ajuda, True, (90, 90, 90)), (10, y_txt + 24))

    def salvar(self, tela):
        pasta = Path("report/figures")
        pasta.mkdir(parents=True, exist_ok=True)
        if self.modo == 3:
            caminho = pasta / "scanline_poligono.png"
        elif self.modo == 4:
            caminho = pasta / f"antialiasing_{self.cena_aa}_ssaa{self.fator}.png"
        elif self.modo == 5:
            alg = "cohen_sutherland" if self.alg_recorte.startswith("Cohen") else "liang_barsky"
            caminho = pasta / f"recorte_{VISTAS_RECORTE[self.vista_recorte]}_{alg}.png"
        elif self.modo == 6:
            suj = "regiao" if self.sujeito.startswith("região") else "u"
            jan = {"retângulo": "retangulo", "hexágono": "hexagono",
                   "L côncavo": "l_concavo"}[self.tipo_janela]
            alg = "sh" if self.alg_poligono.startswith("Sutherland") else "wa"
            caminho = pasta / f"poligono_{alg}_{suj}_{jan}_{VISTAS_RECORTE[self.vista_recorte]}.png"
        else:
            nome = "raster_arvore" if self.modo == 1 else "raster_segmento"
            caminho = pasta / f"{nome}_{self.algoritmo.lower()}.png"
        pygame.image.save(tela, str(caminho))
        print("salvo:", caminho)

    # ------------------------------------------------------------ eventos
    def janela_do_arrasto(self, pos):
        (xa, ya), (xb, yb) = self.arrasto, pixel_do_mouse(pos)
        if xa != xb and ya != yb:
            self.janela = (float(min(xa, xb)), float(min(ya, yb)),
                           float(max(xa, xb)), float(max(ya, yb)))
            self.rasterizar()

    def tratar_clique(self, pos):
        if pos[1] >= LINHAS * CELULA:
            return
        p = pixel_do_mouse(pos)
        if self.modo in (5, 6):
            self.arrasto = p
            return
        if self.modo == 3 and self.editando:
            self.poligono.append(p)
            self.rasterizar()
            return
        if self.modo != 2:
            return
        if self.clique is None:
            self.clique = p
        else:
            self.segmento = (*self.clique, *p)
            self.clique = None
            self.rasterizar()
            if self.algoritmo == "Bresenham":
                imprimir_rastro(*self.segmento)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "data/tree2D_Nterm016.vtk"
    app = App(ler_vtk(caminho))

    pygame.display.init()             # só vídeo e fonte (evita iniciar o áudio)
    pygame.font.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("TP1 CCO - modo didático")
    fonte = pygame.font.Font(None, 22)
    app.rasterizar()

    relogio = pygame.time.Clock()
    quadro = 0
    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                app.tratar_clique(evento.pos)
            elif evento.type == pygame.MOUSEMOTION and app.arrasto is not None:
                app.janela_do_arrasto(evento.pos)
            elif evento.type == pygame.MOUSEBUTTONUP and evento.button == 1:
                if app.arrasto is not None:
                    app.janela_do_arrasto(evento.pos)
                    app.arrasto = None
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                elif evento.key == pygame.K_k and app.modo == 4:
                    app.fator = {2: 4, 4: 8, 8: 2}[app.fator]
                    app.rasterizar()
                elif evento.key == pygame.K_t and app.modo == 4:
                    app.cena_aa = "árvore" if app.cena_aa == "leque" else "leque"
                    app.rasterizar()
                elif evento.key == pygame.K_g and app.modo == 5:
                    app.alg_recorte = ("Liang–Barsky" if app.alg_recorte == "Cohen–Sutherland"
                                       else "Cohen–Sutherland")
                    app.rasterizar()
                elif evento.key == pygame.K_p and app.modo == 6:
                    app.sujeito = ("forma em U" if app.sujeito == "região de interesse"
                                   else "região de interesse")
                    app.rasterizar()
                elif evento.key == pygame.K_j and app.modo == 6:
                    i = TIPOS_JANELA.index(app.tipo_janela)
                    app.tipo_janela = TIPOS_JANELA[(i + 1) % len(TIPOS_JANELA)]
                    app.rasterizar()
                elif evento.key == pygame.K_w and app.modo == 6:
                    app.alg_poligono = ("Weiler–Atherton" if app.alg_poligono == "Sutherland–Hodgman"
                                        else "Sutherland–Hodgman")
                    app.rasterizar()
                elif evento.key == pygame.K_i and app.modo == 6:
                    app.imprimir_recorte_poligono()
                elif evento.key == pygame.K_v and app.modo in (5, 6):
                    app.vista_recorte = (app.vista_recorte + 1) % len(VISTAS_RECORTE)
                    app.rasterizar()
                elif evento.key == pygame.K_i and app.modo == 5:
                    app.imprimir_recorte()
                elif evento.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                    pygame.K_5, pygame.K_6):
                    app.modo = evento.key - pygame.K_0
                    app.clique = None
                    app.anim = None
                    app.rasterizar()
                elif evento.key == pygame.K_n and app.modo == 3:
                    app.poligono, app.editando, app.anim = [], True, None
                    app.rasterizar()
                elif evento.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and app.editando:
                    if len(app.poligono) >= 3:
                        app.editando = False
                        app.rasterizar()
                        imprimir_scanline(app.poligono)
                elif evento.key == pygame.K_SPACE and app.modo == 3 and not app.editando:
                    app.anim = 0
                    app.rasterizar()
                elif evento.key == pygame.K_b:
                    app.algoritmo = "DDA" if app.algoritmo == "Bresenham" else "Bresenham"
                    app.rasterizar()
                elif evento.key == pygame.K_l:
                    app.reta_ideal = not app.reta_ideal
                elif evento.key == pygame.K_s:
                    app.salvar(tela)

        quadro += 1
        if quadro % 3 == 0:                         # animação a ~10 linhas por segundo
            app.avancar_animacao()

        app.desenhar(tela, fonte)
        pygame.display.flip()
        relogio.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
