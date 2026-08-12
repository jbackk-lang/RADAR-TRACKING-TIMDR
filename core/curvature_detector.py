"""
CurvatureDetector -- dyskretna krzywizna trajektorii jako niezależny
(od JRegulatora) sygnał wykrywania manewru, oparty na "THE-GEO PRO"
z repo jbackk-lang/THE.

Pochodzenie i uczciwe zastrzeżenie
-----------------------------------
Repo THE opisuje w pseudokodzie dyskretną krzywiznę i torsję trajektorii
(krzywizna Freneta-Serreta w wersji dyskretnej: kappa = |D_t - D_t-1| / G,
gdzie D to znormalizowany wektor kierunku, G to długość kroku). Sprawdzone
na 4 realnych trasach GPS (data/real_trips_sample.csv, z etykietowaną
prawdziwą zmianą kursu Heading_Change) z dwoma wynikami:

1. Torsja z oryginalnego pseudokodu NIE ma zastosowania w 2D -- torsja
   z definicji mierzy wychodzenie trajektorii poza płaszczyznę ruchu,
   a tor 2D z definicji leży w jednej płaszczyźnie, więc torsja jest tu
   strukturalnie zawsze zero. Pominięta.

2. Kolejna wersja pseudokodu ("THE-GEO PRO 2D->3D") próbowała to obejść
   przez sztuczną "głębokość percepcyjną" dz = f(dx, dy) i nową "torsję"
   tau = dx*dy / G^2. Sprawdzone i ODRZUCONE: tau wychodzi niezerowe dla
   zwykłego ruchu po linii prostej pod kątem do osi (np. dx=3, dy=4 na
   każdym kroku -> tau=0.375 mimo zerowego manewru z definicji), i NIE
   jest niezmiennicze względem obrotu układu współrzędnych -- ten sam
   fizyczny ruch po linii prostej daje różne "tau" w zależności od tego,
   jak akurat narysowano osie X/Y. Dokładnie ta sama kategoria fałszywych
   alarmów co w odrzuconych wcześniej wariantach operator_J. Niewdrożone.

3. Sama krzywizna (kappa) w 2D, LICZONA DOSŁOWNIE wg pseudokodu (bez
   zabezpieczeń), jest bezużyteczna na realnych danych GPS: przy postoju
   lub bardzo wolnym ruchu szum pozycji GPS (ułamki metra) dzielony przez
   mikroskopijny krok G daje ogromne fałszywe piki (test: 71.2 przy czystym
   postoju z szumem GPS ~30cm, zamiast 0). To ta sama kategoria błędu co
   dzielenie przez zero w JRegulatorze (compounding bug) i w Helix-Astro
   (T1) -- tylko tu nie crashuje, tylko wzmacnia szum.

Naprawa: próg minimalnej długości kroku (MIN_STEP_M), poniżej którego
krzywizna nie jest liczona (zwracane 0.0 -- "brak wystarczającego ruchu
żeby ocenić skręt", nie "brak skrętu"). Wartość 3.0 m dobrana empirycznie
z realnych danych (patrz data/validate_curvature_detector.py) -- to punkt,
w którym korelacja krzywizny z prawdziwą zmianą kursu (Heading_Change) jest
najwyższa na wszystkich 4 testowanych trasach.

Wyniki walidacji (Pearson, krzywizna vs |Heading_Change|, 4 realne trasy):
    bez progu (surowy pseudokod):  -0.09 .. -0.004  (bezużyteczne / szum)
    z progiem 3.0 m:                0.47 .. 0.76    (realna korelacja)
Dla porównania: żaden z 6 wariantów operator_J testowanych wcześniej nie
przekroczył ~17% pokrycia górnego decyla z prawdziwymi dużymi |Δv|; ten
detektor osiąga 36-43% pokrycia górnego decyla z prawdziwą zmianą kursu.

Status: to jest DODATKOWY, niezależny sygnał manewru -- nie zastępuje
JRegulatora (który reguluje parametry filtra i bramkuje pomiary) i nie jest
w niego wpięty automatycznie. Można go użyć osobno do np. flagowania
podejrzanych segmentów trajektorii do dalszej analizy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_MIN_STEP_M = 3.0


@dataclass
class CurvatureResult:
    curvature: float          # kappa: 0.0 gdy poniżej progu prędkości lub brak historii
    direction_stability: float  # dot(D_t, D_t-1): 1.0 = idealnie prosto, -1.0 = zawrócenie
    step_m: float              # G_t: długość ostatniego kroku (do diagnostyki)
    gated: bool                 # True jeśli krok był za krótki żeby ocenić kierunek wiarygodnie


class CurvatureDetector:
    """
    Śledzi ostatnie 3 pozycje (x, y) i na bieżąco liczy dyskretną krzywiznę
    trajektorii, z zabezpieczeniem przed wzmacnianiem szumu GPS przy
    postoju/wolnym ruchu (patrz moduł docstring wyżej).

    Użycie:
        det = CurvatureDetector()
        for x, y in trajektoria:
            wynik = det.update(x, y)
            if wynik.curvature > próg:
                ... podejrzany manewr ...
    """

    def __init__(self, min_step_m: float = DEFAULT_MIN_STEP_M):
        if min_step_m < 0:
            raise ValueError("min_step_m nie może być ujemne")
        self.min_step_m = min_step_m
        self._positions: deque[Tuple[float, float]] = deque(maxlen=3)

    def reset(self) -> None:
        self._positions.clear()

    def update(self, x: float, y: float) -> CurvatureResult:
        self._positions.append((x, y))

        if len(self._positions) < 3:
            return CurvatureResult(curvature=0.0, direction_stability=0.0, step_m=0.0, gated=True)

        p_t2, p_t1, p_t = self._positions

        dx_t, dy_t = p_t[0] - p_t1[0], p_t[1] - p_t1[1]
        dx_t1, dy_t1 = p_t1[0] - p_t2[0], p_t1[1] - p_t2[1]

        g_t = math.hypot(dx_t, dy_t)
        g_t1 = math.hypot(dx_t1, dy_t1)

        if g_t < self.min_step_m or g_t1 < self.min_step_m:
            return CurvatureResult(curvature=0.0, direction_stability=0.0, step_m=g_t, gated=True)

        d_t = (dx_t / g_t, dy_t / g_t)
        d_t1 = (dx_t1 / g_t1, dy_t1 / g_t1)

        diff = math.hypot(d_t[0] - d_t1[0], d_t[1] - d_t1[1])
        kappa = diff / g_t

        dir_stability = d_t[0] * d_t1[0] + d_t[1] * d_t1[1]  # dot product, oba wektory jednostkowe

        return CurvatureResult(curvature=kappa, direction_stability=dir_stability, step_m=g_t, gated=False)
