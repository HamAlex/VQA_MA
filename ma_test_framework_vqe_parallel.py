import csv
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np

# ==========================================
# Konfiguration und globale Parameter
# ==========================================

TOTAL_CORES = max(1, multiprocessing.cpu_count() - 2)
THREADS_PER_TASK = 10
TASKS_PARALLEL = max(1, TOTAL_CORES // THREADS_PER_TASK)

csv_filename = "vqe_masterarbeit_results_multicore.csv"
ansatz_arten = ['TwoLocal', 'EfficientSU2']
fehlerraten = np.linspace(0.0, 0.1, 101).tolist()
fehlerarten = ['depolarizing', 'bitflip']
ansatz_tiefen = [1, 2, 4]
max_iterations = 500
shots = 8192

# ==========================================
# Definition der molekularen Operatoren
# ==========================================
def get_h2_molecule():
    """Erstellt den Hamilton-Operator für das H2-Molekül im sto-3g-Basissatz mit Parity-Mapping (2 Qubits)."""
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
    problem = driver.run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    return "H2_2_Qubits", qubit_op

def get_lih_reduced_molecule():
    """Erstellt den Hamilton-Operator für das LiH-Molekül unter Verwendung der Freeze-Core-Approximation (8 Qubits)."""
    driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.546", basis="sto3g")
    problem = driver.run()
    transformer = FreezeCoreTransformer(freeze_core=True)
    problem_reduced = transformer.transform(problem)
    mapper = ParityMapper(num_particles=problem_reduced.num_particles)
    qubit_op = mapper.map(problem_reduced.hamiltonian.second_q_op())
    return "LiH_Reduced_8_Qubits", qubit_op


# ==========================================
# Generierung der Rauschmodelle
# ==========================================
def create_noise_model(error_type, p):
    if p == 0.0:
        return None
    noise_model = NoiseModel()
    if error_type == 'bitflip':
        error_1q = pauli_error([('X', p), ('I', 1 - p)])
        error_2q = error_1q.tensor(error_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    elif error_type == 'depolarizing':
        error_1q = depolarizing_error(p, 1)
        error_2q = depolarizing_error(p, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    return noise_model

# ==========================================
# Worker-Funktion (Für parallele Ausführung)
# ==========================================
def run_vqe_task(task_args):
    """Führt eine einzelne VQE Simulation komplett unabhängig aus."""
    mol_name, qubit_op, exact_energy, tiefe, ansatz_art, opt_name, fehler_art, p = task_args
    start_time = time.time()
    
    try:
        # Optimierer frisch instanziieren
        if opt_name == 'COBYLA':
            optimizer = COBYLA(maxiter=max_iterations)
        else:
            optimizer = SPSA(maxiter=int(max_iterations / 2))
            
        current_noise = create_noise_model(fehler_art, p)
        
        backend_options = {
            "seed_simulator": 42,
            "max_parallel_threads": THREADS_PER_TASK, 
            "max_parallel_experiments": 1,
            "max_parallel_shots": 1
        }
        
        if current_noise:
            backend_options["noise_model"] = current_noise
            
        backend = AerSimulator(**backend_options)
        estimator = BackendEstimatorV2(backend=backend, options={"default_precision": shots ** -0.5})
        
        if ansatz_art == 'TwoLocal':
            ansatz = TwoLocal(qubit_op.num_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=tiefe)
        elif ansatz_art == 'EfficientSU2':
            ansatz = EfficientSU2(qubit_op.num_qubits, reps=tiefe)
        
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        ansatz_isa = pm.run(ansatz)
        qubit_op_isa = qubit_op.apply_layout(ansatz_isa.layout) if ansatz_isa.layout else qubit_op
        
        vqe = VQE(estimator, ansatz_isa, optimizer)
        
        result = vqe.compute_minimum_eigenvalue(operator=qubit_op_isa)
        
        dauer = round(time.time() - start_time, 2)
        energie = result.eigenvalue.real if hasattr(result.eigenvalue, 'real') else result.eigenvalue
        
        return [mol_name, ansatz_art, tiefe, opt_name, fehler_art if p > 0 else "noiseless", p, exact_energy, energie, result.cost_function_evals, dauer]
        
    except Exception as e:
        print(f"Fehler bei der Berechnung: {str(e)}")
        return [mol_name, ansatz_art, tiefe, opt_name, fehler_art if p > 0 else "noiseless", p, exact_energy, "ERROR", 0, 0]


# ==========================================
# Hauptausführung (Simulation)
# ==========================================
if __name__ == '__main__':
    print(f"Gefundene CPU-Kerne: {TOTAL_CORES}")
    print(f"Tasks parallel:      {TASKS_PARALLEL}")
    print(f"Kerne pro Task:      {THREADS_PER_TASK}")
    print(f"Genutzte Kerne:      {TASKS_PARALLEL * THREADS_PER_TASK} / {TOTAL_CORES}")
    print("-" * 50)
    print("Initialisiere molekulare Operatoren...")
    molecules = [get_h2_molecule(), get_lih_reduced_molecule()]

    # Sammle alle Tasks
    tasks = []
    for mol_name, qubit_op in molecules:
        print(f"Berechne klassische Referenz für {mol_name}...")
        exact_solver = NumPyMinimumEigensolver()
        exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real if hasattr(exact_result.eigenvalue, 'real') else exact_result.eigenvalue
        
        for tiefe in ansatz_tiefen:
            for ansatz_art in ansatz_arten:
                for opt_name in ['COBYLA', 'SPSA']:
                    for fehler_art in fehlerarten:
                        for p in fehlerraten:
                            if p == 0.0 and fehler_art == 'bitflip':
                                continue
                            
                            tasks.append((mol_name, qubit_op, exact_energy, tiefe, ansatz_art, opt_name, fehler_art, p))

    total_tasks = len(tasks)
    print(f"\nEs wurden {total_tasks} unabhängige Simulations-Tasks generiert.")
    
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starte Parallelverarbeitung auf {max_workers} CPU-Kernen...\n")

    completed = 0
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Molekuel", "Ansatz_Art", "Ansatz_Tiefe", "Optimierer", "Fehler_Art", "Fehler_Rate", "Exakte_Energie_Hartree", "Berechnete_Energie_Hartree", "Optimierer_Schritte", "Dauer_Sekunden"])
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_vqe_task, task): task for task in tasks}
            
            for future in as_completed(futures):
                completed += 1
                result_row = future.result()
                
                writer.writerow(result_row)
                file.flush()
                
                energie = result_row[7]
                dauer = result_row[9]
                if energie == "ERROR":
                    print(f"[{completed}/{total_tasks}] Abgeschlossen mit Fehler: {result_row[0]} | {result_row[1]} | {result_row[3]} | {result_row[4]} ({result_row[5]})")
                else:
                    print(f"[{completed}/{total_tasks}] Abgeschlossen: {result_row[0]} | {result_row[1]} | {result_row[3]} | {result_row[4]} ({result_row[5]:.3f}) -> {dauer}s | E: {energie:.4f}")

    print("\nSimulationen erfolgreich abgeschlossen. Daten wurden gesichert.")