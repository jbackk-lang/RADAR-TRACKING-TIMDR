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
