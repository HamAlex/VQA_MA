import warnings
from qiskit.circuit.library import TwoLocal, EfficientSU2
import matplotlib.pyplot as plt

print("Generiere hochauflösende Schaltkreis-Bilder...")

# ==========================================
# 1. Das H2-Molekül (2 Qubits) - TwoLocal Ansatz
# ==========================================
# Wir nehmen Tiefe 1 (reps=1), weil das für ein Diagramm am übersichtlichsten ist.
ansatz_h2_twolocal = TwoLocal(num_qubits=2, rotation_blocks='ry', entanglement_blocks='cz', reps=1)

# WICHTIG: .decompose() bricht die "Blackbox" auf und zeigt die echten physikalischen Gates!
fig1 = ansatz_h2_twolocal.decompose().draw(output='mpl', style='clifford')
fig1.savefig('Schaltkreis_H2_TwoLocal_Tiefe1.png', dpi=300, bbox_inches='tight')
print("- H2 TwoLocal (Tiefe 1) gespeichert.")

# ==========================================
# 2. Das LiH-Molekül (4 Qubits) - TwoLocal Ansatz
# ==========================================
ansatz_lih_twolocal = TwoLocal(num_qubits=4, rotation_blocks='ry', entanglement_blocks='cz', reps=1)
# entanglement='linear' (Standard bei TwoLocal) zeigt wunderschön, wie die Qubits nacheinander verschränkt werden
fig2 = ansatz_lih_twolocal.decompose().draw(output='mpl', style='clifford')
fig2.savefig('Schaltkreis_LiH_TwoLocal_Tiefe1.png', dpi=300, bbox_inches='tight')
print("- LiH TwoLocal (Tiefe 1) gespeichert.")

# ==========================================
# 3. Das LiH-Molekül (4 Qubits) - EfficientSU2 Ansatz
# ==========================================
# Um in der Arbeit den Unterschied zwischen den Ansätzen zu zeigen
ansatz_lih_su2 = EfficientSU2(num_qubits=4, reps=1)
fig3 = ansatz_lih_su2.decompose().draw(output='mpl', style='clifford')
fig3.savefig('Schaltkreis_LiH_EfficientSU2_Tiefe1.png', dpi=300, bbox_inches='tight')
print("- LiH EfficientSU2 (Tiefe 1) gespeichert.")

# ==========================================
# 4. Tiefe 2 als Eskalations-Beispiel (H2)
# ==========================================
# Zeigt optisch, wie schnell der Schaltkreis länger (und damit fehleranfälliger) wird
ansatz_h2_tiefe2 = TwoLocal(num_qubits=2, rotation_blocks='ry', entanglement_blocks='cz', reps=2)
fig4 = ansatz_h2_tiefe2.decompose().draw(output='mpl', style='clifford')
fig4.savefig('Schaltkreis_H2_TwoLocal_Tiefe2.png', dpi=300, bbox_inches='tight')
print("- H2 TwoLocal (Tiefe 2) gespeichert.")

print("\nFertig! Alle Bilder liegen jetzt Ordner bereit für LaTeX.")