"""
===============================================================================
VQE Simulation Data Analysis & Statistical Evaluation Framework
===============================================================================
Masterarbeit: Einfluss von Quantenfehlern auf variationelle Quantenalgorithmen
Autor: Alexander Hamann BSc.

Dieses Skript führt eine statistische Auswertung 
der experimentellen VQE-Simulationsdaten (H2 & LiH) durch:
  1. Baseline-Evaluierung (Noiseless, p = 0.0) & Shot-Noise-Charakterisierung
  2. Chemical-Accuracy-Grenzwerte & Rauschkanalanalyse (Bitflip vs. Depolarisierung)
  3. Skalierungsanalyse der Schaltungstiefe (D = 1, 2, 4) & Trade-off-Quantifizierung
  4. Optimierer-Performanz (COBYLA vs. SPSA: MAE, RMSE, Schrittanzahl, False Early Stopping)
  5. Laufzeit- & Komplexitätsskalierung (2 Qubits vs. 8 Qubits / Dichtematrix-Overhead)
  6. Anomalie- & Ausreißerdetektion (z. B. False Early Stopping bei p = 0.041)
===============================================================================
"""

import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Physikalische Konstanten & Schwellenwerte
# =============================================================================
CHEMICAL_ACCURACY_HA: float = 1.6e-3  # 1.6 mHa = 1.0 kcal/mol (Standardgrenze)


@dataclass
class VQERun:
    """Repräsentiert einen einzelnen experimentellen VQE-Simulationslauf."""
    molekuel: str
    ansatz_art: str
    ansatz_tiefe: int
    optimierer: str
    fehler_art: str
    fehler_rate: float
    exakte_energie: float
    berechnete_energie: float
    schritte: int
    dauer_sekunden: float

    @property
    def delta_energie(self) -> float:
        """Absolute Differenz zum exakten quantenmechanischen Grundzustand."""
        return abs(self.berechnete_energie - self.exakte_energie)

    @property
    def signierter_fehler(self) -> float:
        """Signierte Differenz (E_VQE - E_exakt)."""
        return self.berechnete_energie - self.exakte_energie

    @property
    def meets_chemical_accuracy(self) -> bool:
        """Prüft, ob der Lauf innerhalb der chemischen Genauigkeit liegt."""
        return self.delta_energie <= CHEMICAL_ACCURACY_HA


# =============================================================================
# Datenlade- & Validierungsfunktionen
# =============================================================================
def load_vqe_dataset(csv_path: str) -> List[VQERun]:
    """Lädt und parst eine Ergebnis-CSV-Datei in strukturierte Dataclass-Objekte."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Datensatz nicht gefunden: {csv_path}")

    runs: List[VQERun] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                # Behandelt etwaige Statusfehler robust
                if row["Berechnete_Energie_Hartree"] == "ERROR":
                    continue

                run = VQERun(
                    molekuel=row["Molekuel"].strip(),
                    ansatz_art=row["Ansatz_Art"].strip(),
                    ansatz_tiefe=int(row["Ansatz_Tiefe"]),
                    optimierer=row["Optimierer"].strip(),
                    fehler_art=row["Fehler_Art"].strip(),
                    fehler_rate=float(row["Fehler_Rate"]),
                    exakte_energie=float(row["Exakte_Energie_Hartree"]),
                    berechnete_energie=float(row["Berechnete_Energie_Hartree"]),
                    schritte=int(float(row["Optimierer_Schritte"])),
                    dauer_sekunden=float(row["Dauer_Sekunden"]),
                )
                runs.append(run)
            except (ValueError, KeyError) as e:
                print(f"[Warnung] Ungültige Zeile {idx} in {csv_path}: {e}")
    return runs


# =============================================================================
# Statistische Hilfsfunktionen
# =============================================================================
def calc_descriptive_stats(values: List[float]) -> Dict[str, float]:
    """Berechnet deskriptive Standardstatistiken für eine Messreihe."""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    
    n = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if n > 1 else 0.0
    med_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "median": med_val,
        "min": min_val,
        "max": max_val,
    }


def calc_mae_rmse(runs: List[VQERun]) -> Tuple[float, float]:
    """Berechnet Mean Absolute Error (MAE) und Root Mean Square Error (RMSE)."""
    if not runs:
        return 0.0, 0.0
    errors = [r.delta_energie for r in runs]
    mae = statistics.mean(errors)
    rmse = math.sqrt(statistics.mean([e ** 2 for e in errors]))
    return mae, rmse


# =============================================================================
# Analyseschritte
# =============================================================================
def analyze_baseline(runs: List[VQERun], molecule_name: str) -> None:
    """1. Baseline-Check ohne Rauschen (p = 0.0)."""
    noiseless_runs = [r for r in runs if r.fehler_rate == 0.0]
    print("\n" + "=" * 90)
    print(f" 1. BASELINE-ANALYSE (Noiseless, p = 0.0) | {molecule_name}")
    print("=" * 90)
    header = f"{'Ansatz':<15} | {'Tiefe':<5} | {'Optimierer':<8} | {'E_Exakt (Ha)':<14} | {'E_VQE (Ha)':<14} | {'Delta E (Ha)':<12} | {'Chem. Acc.?':<10}"
    print(header)
    print("-" * len(header))
    
    for r in noiseless_runs:
        acc_str = "JA" if r.meets_chemical_accuracy else "NEIN"
        print(f"{r.ansatz_art:<15} | {r.ansatz_tiefe:<5} | {r.optimierer:<8} | {r.exakte_energie:<14.6f} | {r.berechnete_energie:<14.6f} | {r.delta_energie:<12.6f} | {acc_str:<10}")


def analyze_chemical_accuracy_and_noise_channels(runs: List[VQERun], molecule_name: str) -> None:
    """2. Rauschkanalvergleich & Verlust der Chemical Accuracy."""
    print("\n" + "=" * 90)
    print(f" 2. RAUSCHKANAL-VERGLEICH & CHEMICAL ACCURACY | {molecule_name}")
    print("=" * 90)

    for fehler in ["bitflip", "depolarizing"]:
        f_runs = [r for r in runs if r.fehler_art == fehler and r.fehler_rate > 0.0]
        if not f_runs:
            continue
        
        mae, rmse = calc_mae_rmse(f_runs)
        acc_count = sum(1 for r in f_runs if r.meets_chemical_accuracy)
        total = len(f_runs)
        
        # Finde erste Fehlerrate, ab der Chemical Accuracy überschritten wird
        f_runs_sorted = sorted(f_runs, key=lambda x: x.fehler_rate)
        p_loss: Optional[float] = None
        for r in f_runs_sorted:
            if not r.meets_chemical_accuracy:
                p_loss = r.fehler_rate
                break
                
        print(f"\n--- Rauschkanal: {fehler.upper()} ---")
        print(f"  * Stichprobengröße (Noisy Runs) : {total}")
        print(f"  * Mean Absolute Error (MAE)     : {mae:.6f} Ha")
        print(f"  * Root Mean Square Error (RMSE) : {rmse:.6f} Ha")
        print(f"  * Einhaltung Chemical Accuracy  : {acc_count}/{total} ({acc_count/total*100:.2f} %)")
        print(f"  * Verlust Chemical Accuracy ab  : p = {p_loss if p_loss is not None else 'Nie':.4f}")

        # Spezifische Aufschlüsselung nach Tiefe 1
        t1_runs = [r for r in f_runs if r.ansatz_tiefe == 1]
        t1_mae = statistics.mean([r.delta_energie for r in t1_runs])
        print(f"  * MAE bei Schaltungstiefe 1     : {t1_mae:.6f} Ha")


def analyze_depth_scaling(runs: List[VQERun], molecule_name: str) -> None:
    """3. Skalierung der Schaltungstiefe (Trade-off & Rauschakkumulation)."""
    print("\n" + "=" * 90)
    print(f" 3. SKALIERUNGSANALYSE DER SCHALTUNGSTIEFE (D in [1, 2, 4]) | {molecule_name}")
    print("=" * 90)
    
    header = f"{'Tiefe D':<8} | {'Fehlerart':<12} | {'MAE (Ha)':<12} | {'RMSE (Ha)':<12} | {'Max Delta (Ha)':<14} | {'MAE @ p=0.01':<14}"
    print(header)
    print("-" * len(header))

    for d in [1, 2, 4]:
        for fehler in ["depolarizing", "bitflip"]:
            sub = [r for r in runs if r.ansatz_tiefe == d and r.fehler_art == fehler and r.fehler_rate > 0.0]
            if not sub:
                continue
            mae, rmse = calc_mae_rmse(sub)
            max_delta = max(r.delta_energie for r in sub)
            
            p01_runs = [r for r in sub if math.isclose(r.fehler_rate, 0.01, abs_tol=1e-5)]
            mae_p01 = statistics.mean([r.delta_energie for r in p01_runs]) if p01_runs else 0.0
            
            print(f"{d:<8} | {fehler:<12} | {mae:<12.6f} | {rmse:<12.6f} | {max_delta:<14.6f} | {mae_p01:<14.6f}")


def analyze_optimizer_performance(runs: List[VQERun], molecule_name: str) -> None:
    """4. Optimierervergleich: COBYLA vs. SPSA (Genauigkeit, Schritte & Laufzeit)."""
    print("\n" + "=" * 90)
    print(f" 4. OPTIMIERERVERGLEICH (COBYLA vs. SPSA) | {molecule_name}")
    print("=" * 90)

    for opt in ["COBYLA", "SPSA"]:
        sub = [r for r in runs if r.optimierer == opt and r.fehler_rate > 0.0]
        if not sub:
            continue
        
        mae, rmse = calc_mae_rmse(sub)
        times = [r.dauer_sekunden for r in sub]
        steps = [r.schritte for r in sub]
        
        t_stats = calc_descriptive_stats(times)
        s_stats = calc_descriptive_stats(steps)
        
        print(f"\n--- Optimierer: {opt} ---")
        print(f"  * Gesamtanzahl Runs          : {len(sub)}")
        print(f"  * MAE über alle Rauschläufe  : {mae:.6f} Ha")
        print(f"  * RMSE über alle Rauschläufe : {rmse:.6f} Ha")
        print(f"  * Schritte (Mittel ± Std)    : {s_stats['mean']:.1f} ± {s_stats['std']:.1f} (Median: {s_stats['median']:.1f}, Min: {s_stats['min']}, Max: {s_stats['max']})")
        print(f"  * Laufzeit (Mittel ± Std)    : {t_stats['mean']:.2f} s ± {t_stats['std']:.2f} s (Gesamt: {sum(times)/3600:.2f} h)")


def detect_anomalies_and_early_stopping(runs: List[VQERun], molecule_name: str) -> None:
    """5. Detektion spezifischer Anomalien, Peaks & False Early Stopping."""
    print("\n" + "=" * 90)
    print(f" 5. ANOMALIEN & FALSE EARLY STOPPING DETEKTION | {molecule_name}")
    print("=" * 90)
    
    # Suche nach COBYLA Läufen mit verfrühtem Abbruch (< 100 Schritte) bei signifikantem Rauschen
    early_stops = [
        r for r in runs 
        if r.optimierer == "COBYLA" and r.fehler_rate >= 0.03 and r.schritte < 100
    ]
    print(f"Identifizierte COBYLA-Läufe mit verfrühtem Abbruch (Schritte < 100 bei p >= 0.03): {len(early_stops)}")
    
    # Spezifische Analyse des bekannten H2-Peaks bei p = 0.041
    if molecule_name == "H2":
        p_targets = [0.040, 0.041, 0.042]
        print("\n--- Fallstudie: H2 TwoLocal Tiefe 2 Depolarisierend um p = 0.041 ---")
        for p_val in p_targets:
            match = [
                r for r in runs 
                if r.ansatz_art == "TwoLocal" and r.ansatz_tiefe == 2 
                and r.optimierer == "COBYLA" and r.fehler_art == "depolarizing"
                and math.isclose(r.fehler_rate, p_val, abs_tol=1e-4)
            ]
            for m in match:
                print(f"  p = {m.fehler_rate:.3f} -> Berechnet: {m.berechnete_energie:.6f} Ha | Delta E: {m.delta_energie:.6f} Ha | Schritte: {m.schritte} | Dauer: {m.dauer_sekunden:.2f} s")


def compare_molecule_complexity(runs_h2: List[VQERun], runs_lih: List[VQERun]) -> None:
    """6. Komplexitätssprung: 2 Qubits (H2) vs. 8 Qubits (LiH)."""
    print("\n" + "=" * 90)
    print(" 6. SYSTEMKOMPLEXITÄT & SKALIERUNGSSPRUNG (H2 [2 Qubits] vs. LiH [8 Qubits])")
    print("=" * 90)

    # MAE bei p = 0.01 Depolarisierung, Tiefe 1, SPSA
    h2_p01 = [
        r for r in runs_h2 
        if r.ansatz_tiefe == 1 and r.optimierer == "SPSA" 
        and r.fehler_art == "depolarizing" and math.isclose(r.fehler_rate, 0.01, abs_tol=1e-5)
    ]
    lih_p01 = [
        r for r in runs_lih 
        if r.ansatz_tiefe == 1 and r.optimierer == "SPSA" 
        and r.fehler_art == "depolarizing" and math.isclose(r.fehler_rate, 0.01, abs_tol=1e-5)
    ]

    h2_mae = statistics.mean([r.delta_energie for r in h2_p01]) if h2_p01 else 0.0
    lih_mae = statistics.mean([r.delta_energie for r in lih_p01]) if lih_p01 else 0.0

    print(f"  * MAE @ p = 0.01 (Tiefe 1, SPSA, Depol) H2  : {h2_mae:.6f} Ha")
    print(f"  * MAE @ p = 0.01 (Tiefe 1, SPSA, Depol) LiH : {lih_mae:.6f} Ha")
    if h2_mae > 0:
        print(f"  * Multiplikativer Fehleranstieg (LiH / H2) : {lih_mae / h2_mae:.2f}x")

    # Mittlere Laufzeiten pro VQE-Lauf
    h2_mean_t = statistics.mean([r.dauer_sekunden for r in runs_h2])
    lih_mean_t = statistics.mean([r.dauer_sekunden for r in runs_lih])
    print(f"\n  * Mittlere Laufzeit pro Lauf H2            : {h2_mean_t:.2f} s")
    print(f"  * Mittlere Laufzeit pro Lauf LiH           : {lih_mean_t:.2f} s ({lih_mean_t/60:.1f} min)")
    print(f"  * Laufzeit-Skalierungsfaktor (LiH / H2)    : {lih_mean_t / h2_mean_t:.1f}x")


# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    h2_path = os.path.join("sim_data", "vqe_masterarbeit_results_H2.csv")
    lih_path = os.path.join("sim_data", "vqe_masterarbeit_results_LiH8.csv")

    print("===============================================================================")
    print(" VQE EXPERIMENT STATISTICAL EVALUATION PIPELINE")
    print("===============================================================================")
    
    runs_h2 = load_vqe_dataset(h2_path)
    runs_lih = load_vqe_dataset(lih_path)
    
    print(f"Geladene Datensätze: H2 ({len(runs_h2)} Zeilen), LiH ({len(runs_lih)} Zeilen)")

    # 1. H2 Analysen
    analyze_baseline(runs_h2, "H2 (2 Qubits)")
    analyze_chemical_accuracy_and_noise_channels(runs_h2, "H2 (2 Qubits)")
    analyze_depth_scaling(runs_h2, "H2 (2 Qubits)")
    analyze_optimizer_performance(runs_h2, "H2 (2 Qubits)")
    detect_anomalies_and_early_stopping(runs_h2, "H2 (2 Qubits)")

    # 2. LiH Analysen
    analyze_baseline(runs_lih, "LiH (8 Qubits)")
    analyze_chemical_accuracy_and_noise_channels(runs_lih, "LiH (8 Qubits)")
    analyze_depth_scaling(runs_lih, "LiH (8 Qubits)")
    analyze_optimizer_performance(runs_lih, "LiH (8 Qubits)")
    detect_anomalies_and_early_stopping(runs_lih, "LiH (8 Qubits)")

    # 3. Komplexitätssprung
    compare_molecule_complexity(runs_h2, runs_lih)

    print("\n" + "=" * 90)
    print(" STATISTISCHE EVALUIERUNG ERFOLGREICH ABGESCHLOSSEN")
    print("=" * 90)


if __name__ == "__main__":
    main()
