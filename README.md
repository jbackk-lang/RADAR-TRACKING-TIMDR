# RADAR-CUSTOM-TIMDR

Osobny, prosty projekt (strategy pattern): `RadarTrackerCustom(backend)` wybiera
filtr (`kalman` lub `particle`) i deleguje do niego `.update(measurement)`.

```python
from core.radar_tracker_custom import RadarTrackerCustom

tracker = RadarTrackerCustom("kalman")
for m in [10, 12, 15, 14, 13]:
    print(tracker.update(m))
```

To jest osobny projekt od RADAR-TRACKING (nie nadpisuje go). Struktura plików
(`core/radar_tracker.py` itd.) pokrywa się nazwami, ale to inny, prostszy
tracker jednowymiarowy/skalarny, a nie wielobiektowy tracker 2D z RADAR-TRACKING.

## Filtry

- **`KalmanFilterCustom`** — 1D filtr Kalmana ze stałą prędkością (stan:
  pozycja + prędkość). Naprawiony błąd: `float(self.x[0])` na wycinku tablicy
  (1,) zamiast skalara — numpy dziś zgłasza na to `DeprecationWarning`, w
  przyszłej wersji będzie to błąd. Naprawione przez `self.x[0, 0]`.
- **`ParticleFilterCustom`** — filtr cząsteczkowy (bootstrap/SIR).

## Podsumowanie w trzech zdaniach (uczciwe)

Filtr cząsteczkowy przeszedł dwie rundy poprawek: pierwsza (`process_std=4.0`,
`measurement_sigma=5.0`) wyglądała na naprawioną, bo działała na syntetycznym
teście (v=3/krok), ale na prawdziwych danych GPS (prędkości do 33 m/s) błąd
rósł katastrofalnie (średnio ~2037m, szczyty >5500m) — była to poprawka
dopasowana do zbyt wolnego scenariusza testowego, nie naprawa struktury.
Druga runda dodała cząsteczkom stan prędkości (tak jak w filtrze Kalmana),
co ustabilizowało wynik na wszystkich 4 prawdziwych trasach GPS (średni błąd
0.85–7.73m, patrz tabela niżej) — jeden lokalny skok do 258.84m na trasie
T-29 to prawdziwa gwałtowna zmiana prędkości, po której filtr wraca do
błędu ~1.6m, a nie trwała utrata śledzenia. Domyślne parametry
(`vel_process_std`, `pos_process_std`, `measurement_sigma`, `process_var`,
`measurement_var`) są dobrane pod te konkretne dane walidacyjne — przy innej
dynamice celu / innym szumie sensora wymagają ponownego strojenia, tak samo
jak `min_speed` w repo RADAR-TRACKING.
 „Uwaga: parametry filtra cząsteczkowego wymagają strojenia pod dane”.
 
## Walidacja na prawdziwych danych

`data/real_trips_sample.csv` — te same 4 prawdziwe trasy GPS co w
RADAR-TRACKING (źródło: `sobhan-moosavi/Trajectory_Segmentation` na GitHubie).

```
python3 data/validate_on_real_trips.py
```

| trasa | backend  | n   | mean   | max     | last20 |
|-------|----------|-----|--------|---------|--------|
| T-1   | kalman   | 946 | 7.78m  | 29.22m  | 0.97m  |
| T-1   | particle | 946 | 2.03m  | 15.61m  | 0.62m  |
| T-14  | kalman   | 953 | 6.51m  | 29.56m  | 8.40m  |
| T-14  | particle | 953 | 1.78m  | 17.44m  | 1.79m  |
| T-29  | kalman   | 956 | 5.27m  | 43.12m  | 5.18m  |
| T-29  | particle | 956 | 7.73m  | 258.84m | 1.64m  |
| T-3   | kalman   | 944 | 3.43m  | 21.21m  | 3.23m  |
| T-3   | particle | 944 | 0.85m  | 8.01m   | 0.68m  |

## Testy

```
python3 -m pytest tests/ -v
```

12 testów, wszystkie przechodzą, w tym testy regresyjne na oba opisane wyżej
błędy (numpy deprecation w Kalmanie, katastrofalna dywergencja w filtrze
cząsteczkowym na prawdziwych danych).
Filtr cząsteczkowy śledzi obiekt, trzymając tysiące "hipotez" (cząsteczek) —
każda to zgadywanka "obiekt może być tutaj". Po każdym pomiarze cząsteczki
bliższe pomiarowi dostają większą wagę, dalsze mniejszą, i filtr losuje nowy
zestaw cząsteczek faworyzując te z wyższą wagą. Średnia z cząsteczek to
estymata pozycji.

Problem: między pomiarami cząsteczki muszą się same "rozproszyć", żeby
nadążyć za ruchem obiektu — to jak grupa ludzi z zawiązanymi oczami, która co
sekundę robi mały losowy krok, próbując trafić tam, gdzie faktycznie jest
szukana osoba. Jeśli osoba idzie wolno, mały losowy krok wystarczy. Ale jeśli
osoba biegnie (jak samochód na autostradzie, do 33 m/s), mały losowy krok
nigdy jej nie dogoni — wszyscy zostają z tyłu. Wtedy WSZYSTKIE cząsteczki
dostają fatalną wagę naraz, bo żadna nie jest blisko prawdy. Filtr traci
punkt odniesienia: zamiast poprawiać się z każdym pomiarem, zamraża się w
miejscu, podczas gdy prawdziwy obiekt ucieka coraz dalej. To jest właśnie
"katastrofalna dywergencja" — błąd nie maleje ani nie oscyluje, tylko rośnie
bez końca, aż filtr staje się bezużyteczny.

Pierwsza próba naprawy po prostu kazała cząsteczkom robić większe losowe
kroki (`process_std` z 1.0 na 4.0). To zadziałało w teście, w którym obiekt
poruszał się wolno — ale to był zbyt łagodny test. Przy prawdziwej prędkości
samochodu i tak było za wolno: średni błąd wynosił ponad 2 kilometry.

Prawdziwa naprawa: zamiast kazać cząsteczkom "chodzić losowo szybciej", dano
im pamięć prędkości — każda cząsteczka wie nie tylko "gdzie jestem", ale też
"jak szybko się poruszam", i przewiduje swoją następną pozycję na tej
podstawie (dokładnie tak, jak robi to filtr Kalmana). To jak zamiana ludzi z
zawiązanymi oczami robiących losowe kroki na ludzi, którzy słyszą, w którą
stronę i jak szybko biegnie szukana osoba, i biegną w tym samym kierunku,
zamiast błądzić na oślep. Po tej zmianie filtr nadąża nawet przy prędkościach
autostradowych — patrz tabela z prawdziwymi danymi wyżej.

## Topologiczny regulator filtra (Jop / twist / defect / resonance)

`core/j_regulator.py` -- adaptuje parametry filtra na podstawie pary
kolejnych ech radaru, wg pomysłu:

```python
Jop = J(twist, defect, resonance)
j = Jop(prev_echo, curr_echo)
if j["defect"] > DEFECT_THRESHOLD:
    return None  # odrzucamy pomiar
```

Zastrzeżenie: operatory J/M (twist), ΔS (defekt), R (rezonans) w
GIA-and-TIMDR (`core/operators.py`) są zdefiniowane dla ciągów bajtów
(kompresja), nie dla skalarnych ech radaru. `JRegulator` to nowa, ciągła
adaptacja tamtej idei (M = orientacja zmiany, ΔS = gwałtowność zmiany,
R = energia), nie przeniesienie gotowego kodu. `resonance` jako "energia"
wymaga pamięci w czasie, więc `JRegulator` jest klasą ze stanem (EMA), a
nie czystą funkcją `Jop(prev, curr)` jak w pseudokodzie.

`gate()` (wersja minimalna) jest podpięta pod backend `kalman` -- odrzucony
pomiar powoduje krok `predict()` (coasting na modelu ruchu) zamiast
`update(z)`. `regulate_particle_params()` (wersja cząsteczkowa) jest
podpięta pod backend `particle` -- podbija `process_std` przy dużym twist,
`measurement_sigma` przy dużym defect, i wywołuje `roughen()` (rozproszenie
chmury, nazwa `spread` była już zajęta jako właściwość) przy dużym
resonance. Domyślnie `regulator=None` -- zachowanie identyczne jak wcześniej.

```python
from core.radar_tracker_custom import RadarTrackerCustom
from core.j_regulator import JRegulator

tracker = RadarTrackerCustom("particle", regulator=JRegulator(), seed=0)
tracker.update(measurement)
```

### Uczciwy wynik walidacji (`python3 data/validate_regulator.py`)

Po drodze znalazłem i naprawiłem realny błąd kompozycji: pierwsza wersja
skalowała parametry od ich AKTUALNEJ (już podbitej) wartości zamiast od
stałej bazy, więc na danych, gdzie regulator wyzwalał się niemal co krok,
`process_std` rosło mnożnikowo przez 956 kroków -- estymaty rzędu 1e20+ m.
Naprawione przez skalowanie zawsze od wartości bazowych z `__init__`.

**Wersja kalmanowa (gate):** realne, normalne przyspieszenia w danych GPS
dają skoki 4-32m/krok -- to nie anomalie, to zwykła jazda. Przy zbyt niskim
progu (15) regulator odrzucał 30-50% prawdziwych pomiarów i błąd średni
rósł ze 3-8m do 140-250m -- realne pogorszenie. Przy progu ≥35 (obecna
wartość domyślna) regulator nigdy się nie wyzwala na tych 4 trasach --
bezpieczny no-op, zweryfikowany testem jednostkowym z syntetycznym wyrzutem,
że faktycznie łapie prawdziwe anomalie, gdy się pojawią. Nie mam na tych
danych przykładu, gdzie wersja kalmanowa realnie POMAGA -- tylko dowód, że
przy złym progu szkodzi, a przy bezpiecznym nic nie robi.

**Wersja cząsteczkowa pomaga, zmierzone:** średni błąd na 4 trasach spada
z 3.10m (bez regulatora) do 1.04m (z pełnym regulatorem), najgorszy
przypadek (T-29) z 258.84m do 12.38m. Ablacja pokazuje, że nie wszystkie
trzy mechanizmy wnoszą tyle samo -- wyłączenie samego podbicia
`measurement_sigma` po defekcie daje jeszcze lepszy wynik (0.83m / 8.36m)
niż pełna wersja. Zostawiony w kodzie z domyślną wartością zamiast dalej
dostrajany pod te same 4 trasy, żeby nie powtórzyć przeuczenia z Rundy 1
filtra cząsteczkowego.

| wariant particle                          | avg mean | worst max |
|--------------------------------------------|---------:|----------:|
| plain (bez regulatora)                      |   3.10m  |  258.84m  |
| pełny regulator                             |   1.04m  |   12.38m  |
| bez roughening                              |   1.37m  |   19.14m  |
| bez sigma-boost                             |   0.83m  |    8.36m  |
| bez process-boost                           |   1.36m  |   23.49m  |

25 testów (`python3 -m pytest tests/ -v`), wszystkie przechodzą, w tym
regresyjny test na błąd kompozycji opisany wyżej.

### Druga symulacja: wstrzyknięte usterki sensora (`data/validate_regulator_glitch_injection.py`)

Powyższa walidacja pokazała, że na czystych danych GPS wersja kalmanowa
(gate) nigdy się nie wyzwala -- bo tam nie ma prawdziwych pojedynczych
anomalii, tylko szybka, legalna jazda. Żeby to sprawdzić uczciwie, druga
symulacja wstrzykuje symulowane usterki sensora (pojedynczy pomiar
przesunięty o 150-400m, jak odbicie wielodrogowe GPS) co 40 kroków na
każdej z 4 prawdziwych tras, i liczy błąd względem prawdziwej (nieuszkodzonej)
pozycji.

| trasa | backend  | bez regulatora | z regulatorem | poprawa |
|-------|----------|----------------:|---------------:|--------:|
| T-1   | kalman   | 14.02m | 8.12m | 42.1% |
| T-1   | particle | 2.10m | 0.83m | 60.2% |
| T-14  | kalman   | 14.14m | 6.79m | 52.0% |
| T-14  | particle | 1.89m | 0.79m | 58.3% |
| T-29  | kalman   | 12.28m | 5.47m | 55.4% |
| T-29  | particle | 2082.99m | 0.84m | 100.0% |
| T-3   | kalman   | 10.95m | 3.60m | 67.1% |
| T-3   | particle | 0.88m | 0.65m | 26.3% |

Tym razem obie wersje regulatora realnie pomagają -- bo tym razem jest
coś do złapania. Na T-29 czysty filtr cząsteczkowy (bez regulatora) sam
katastrofalnie się rozjeżdża pod wpływem jednej dużej usterki (2083m
średniego błędu) -- z regulatorem wraca do 0.84m.

Sprawdzone też, jak duża musi być usterka, żeby regulator faktycznie
pomógł (kalman, T-1): przy usterkach powyżej progu `defect_threshold=35`
regulator wyraźnie pomaga; przy usterkach w okolicach lub poniżej progu
(30-60m i mniej) jest neutralny albo lekko szkodliwy (jeden niefortunny
gate na normalnym, nie-anomalnym pomiarze). To zgodne z tym, jak próg
działa z definicji -- nie jest to nowy problem, tylko potwierdzenie
zakresu działania.

26 testów łącznie (`python3 -m pytest tests/ -v`), w tym regresyjny test
na ten scenariusz z wstrzykniętą usterką.
