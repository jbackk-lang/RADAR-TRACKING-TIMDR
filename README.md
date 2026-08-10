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
autostradowych — patrz tabela z prawdziwymi danymi niżej.
