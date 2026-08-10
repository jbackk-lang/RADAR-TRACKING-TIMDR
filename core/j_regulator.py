"""
JRegulator -- "topologiczny regulator filtra": adaptuje parametry filtra
na podstawie pary kolejnych ech radaru (prev_echo, curr_echo), tak jak w
Twoim pseudokodzie:

    Jop = J(twist, defect, resonance)
    j = Jop(prev_echo, curr_echo)
    if j["defect"] > DEFECT_THRESHOLD: return None  # odrzucamy pomiar

Uczciwe zastrzeżenie na start: w GIA-and-TIMDR (core/operators.py) operatory
J/M (twist), ΔS (defekt) i R (rezonans) są zdefiniowane dla ciągów BAJTÓW
(kompresja), nie dla pojedynczych skalarnych ech radaru. Nie ma tam gotowej
wersji działającej na dwóch liczbach zmiennoprzecinkowych, więc `twist`,
`defect` i `resonance` poniżej to NOWA, ciągła adaptacja tamtych operatorów,
zbudowana na tej samej idei (M = orientacja zmiany, ΔS = gwałtowność zmiany,
R = energia), a nie przeniesienie istniejącego, przetestowanego kodu.
Analogia jest zamierzona i sensowna, ale to inna implementacja.

Definicje:
    twist(prev, curr)     = curr - prev
                             (podpisana zmiana -- odpowiednik M/op_J:
                             "orientacja zmiany" z operators.py, przeniesiona
                             z XOR bajtów na różnicę skalarów)

    defect(prev, curr)    = |twist|
                             (wielkość skoku -- odpowiednik ΔS/op_deltaS:
                             tam próg na |tau[i]-tau[i-1]|, tu wprost na
                             wielkość skoku echa)

    resonance             = wygładzona (EMA) wielkość ostatnich defektów
                             (odpowiednik R/op_R: "energia skrętu" -- ale
                             energia to zjawisko czasowe, nie chwilowe, więc
                             wymaga pamięci; dlatego JRegulator jest klasą ze
                             stanem, a nie czystą funkcją Jop(prev, curr) z
                             Twojego pseudokodu)

To jest realny, znany w literaturze filtrowania pomysł pod inną nazwą:
"innovation-based adaptive filtering" (skalowanie szumu procesu/pomiaru na
podstawie wielkości reszty pomiarowej) + "roughening" / anti-deprivation
resampling dla filtra cząsteczkowego.

UCZCIWY WYNIK WALIDACJI NA PRAWDZIWYCH DANYCH (data/validate_regulator.py),
bo pierwsza wersja miała dwa realne problemy, nie tylko kwestię strojenia:

1. BŁĄD KOMPOZYCJI (naprawiony): pierwsza wersja odczytywała aktualne
   (być może już podbite) parametry filtra i mnożyła je znowu przy każdym
   wyzwoleniu regulatora. Na prawdziwych danych GPS regulator wyzwalał się
   niemal na każdym kroku, więc process_std rosło mnożnikowo (~1.5x) przez
   956 kroków z rzędu -- wynik: estymaty rzędu 1e20+ metrów. Naprawione
   przez trzymanie stałych wartości bazowych (core/radar_tracker_custom.py)
   i skalowanie zawsze od nich, nie od aktualnego stanu.

2. ŹLE DOBRANY PRÓG DEFEKTU DLA WERSJI KALMANA (częściowo rozwiązany):
   prawdziwe, normalne przyspieszenia w danych GPS dają skoki pozycji
   4-32m na krok (patrz rozkład w data/validate_regulator.py) -- to nie
   są anomalie, to zwykła jazda. Przy DEFECT_THRESHOLD=15 (pierwotna
   wartość) regulator odrzucał 30-50% prawdziwych pomiarów jako "defekt" i
   każdy odrzucony pomiar pogarszał wynik (coasting bez korekty), dając
   błąd średni 140-250m zamiast 3-8m bez regulatora -- realne pogorszenie,
   nie usprawnienie. Przy progu ≥35 (obecna wartość domyślna, ustawiona
   powyżej maksymalnego zaobserwowanego skoku w tych 4 trasach) regulator
   Kalmana nigdy się nie wyzwala na tych danych -- czyli jest bezpiecznym
   no-op, zweryfikowanym testem jednostkowym z syntetycznym wyrzutem (skok
   o 490), że faktycznie łapie prawdziwe anomalie, gdy takie się pojawią.
   Innymi słowy: NIE mam w tym zbiorze danych przykładu, na którym wersja
   kalmanowa regulatora realnie POMAGA -- tylko dowód, że przy złym progu
   szkodzi, i że przy bezpiecznym progu nic nie robi.

3. WERSJA DLA FILTRA CZĄSTECZKOWEGO POMAGA, zmierzone uczciwie: średni
   błąd na 4 trasach spada z 3.10m (bez regulatora) do 1.04m (z pełnym
   regulatorem), a najgorszy przypadek (T-29) z 258.84m do 12.38m. Ablacja
   (data/validate_regulator.py) pokazuje, że NIE wszystkie trzy mechanizmy
   wnoszą tyle samo: wyłączenie samego podbicia measurement_sigma po
   defekcie (zostawiając tylko podbicie process_std po twist i roughening
   po resonance) daje jeszcze lepszy wynik (0.83m / 8.36m) niż pełna wersja
   -- czyli measurement_sigma_boost może być tu neutralne albo lekko
   szkodliwe. Zostawiony w kodzie z domyślną wartością 2.0 zamiast
   dostrajany dalej pod te same 4 trasy, żeby nie powtórzyć błędu z Rundy 1
   filtra cząsteczkowego (przeuczenie pod mały zbiór walidacyjny).

Progi (twist_threshold, defect_threshold, resonance_threshold) są punktem
startowym dobranym pod data/real_trips_sample.csv, nie uniwersalną stałą --
przy innej dynamice celu wymagają ponownej walidacji, tak jak min_speed w
TIMDR-T.
"""
from __future__ import annotations

import numpy as np


class JRegulator:
    def __init__(
        self,
        twist_threshold: float = 8.0,
        defect_threshold: float = 35.0,
        resonance_threshold: float = 10.0,
        resonance_smoothing: float = 0.15,
        process_std_boost: float = 1.5,
        measurement_sigma_boost: float = 2.0,
    ):
        self.twist_threshold = twist_threshold
        self.defect_threshold = defect_threshold
        self.resonance_threshold = resonance_threshold
        self.resonance_smoothing = resonance_smoothing
        self.process_std_boost = process_std_boost
        self.measurement_sigma_boost = measurement_sigma_boost

        self._prev_echo: float | None = None
        self._resonance_ema: float = 0.0

    def step(self, curr_echo: float) -> dict:
        """Zwraca {"twist", "defect", "resonance"} dla pary
        (poprzednie echo, curr_echo) i przesuwa stan o jeden krok.
        Pierwsze wywołanie (brak poprzedniego echa) zwraca zera --
        nie ma z czego policzyć zmiany."""
        if self._prev_echo is None:
            self._prev_echo = curr_echo
            return {"twist": 0.0, "defect": 0.0, "resonance": 0.0}

        twist = curr_echo - self._prev_echo
        defect = abs(twist)

        alpha = self.resonance_smoothing
        self._resonance_ema = alpha * defect + (1 - alpha) * self._resonance_ema

        self._prev_echo = curr_echo
        return {"twist": twist, "defect": defect, "resonance": self._resonance_ema}

    def gate(self, curr_echo: float) -> float | None:
        """Wersja minimalna z Twojego pseudokodu: odrzuca pomiar (zwraca
        None), jeśli defekt przekracza próg. Nie przesuwa stanu resonance
        podwójnie -- gate() i regulate_particle_params() dzielą to samo
        step(), więc wołaj tylko jedno z nich per krok, nie oba."""
        j = self.step(curr_echo)
        if j["defect"] > self.defect_threshold:
            return None
        return curr_echo

    def regulate_particle_params(
        self,
        curr_echo: float,
        vel_process_std: float,
        pos_process_std: float,
        measurement_sigma: float,
    ) -> tuple[float, float, float, bool]:
        """Wersja dla filtra cząsteczkowego z Twojego pseudokodu:
        zwraca (nowy vel_process_std, nowy pos_process_std,
        nowy measurement_sigma, should_spread).

        - twist > próg  -> filtr "spodziewa się" manewru, zwiększ szum
          procesu (particles mają nadążyć za gwałtowną zmianą)
        - defect > próg -> pomiar wygląda na wartość odstającą, zaufaj mu
          mniej (zwiększ measurement_sigma zamiast go odrzucać -- łagodniej
          niż gate())
        - resonance > próg -> utrzymująca się seria dużych zmian, nie
          pojedynczy skok -- to sygnał, że chmura cząsteczek może tracić
          namiar; każ ją rozproszyć (patrz ParticleFilterCustom.roughen()
          -- nazwa `spread` z Twojego pseudokodu jest już zajęta w tej
          klasie jako właściwość zwracająca odchylenie standardowe chmury,
          więc metoda rozpraszająca nazywa się inaczej, żeby nie było
          kolizji nazw)
        """
        j = self.step(curr_echo)

        new_vel_std = vel_process_std
        new_pos_std = pos_process_std
        new_sigma = measurement_sigma
        should_spread = False

        if j["twist"] > self.twist_threshold or -j["twist"] > self.twist_threshold:
            new_vel_std = vel_process_std * self.process_std_boost
            new_pos_std = pos_process_std * self.process_std_boost

        if j["defect"] > self.defect_threshold:
            new_sigma = measurement_sigma * self.measurement_sigma_boost

        if j["resonance"] > self.resonance_threshold:
            should_spread = True

        return new_vel_std, new_pos_std, new_sigma, should_spread
