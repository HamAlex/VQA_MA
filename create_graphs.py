import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten laden
df = pd.read_csv('sim_data/vqe_masterarbeit_results_H2.csv')

# 2. Daten filtern (H2, TwoLocal, Tiefe 1, COBYLA)
df_filtered = df[(df['Molekuel'] == 'H2_2_Qubits') & 
                 (df['Ansatz_Tiefe'] == 1) & 
                 (df['Ansatz_Art'] == 'TwoLocal') & 
                 (df['Optimierer'] == 'COBYLA')].copy()

# 3. Noiseless-Werte auf beide Fehlerarten duplizieren, damit die Linien bei 0 starten
df_noiseless = df_filtered[df_filtered['Fehler_Rate'] == 0.0]
exact_energy = df_noiseless['Exakte_Energie_Hartree'].iloc[0]
noiseless_energy = df_noiseless['Berechnete_Energie_Hartree'].iloc[0]

# Nur Rauschen filtern
df_noisy = df_filtered[df_filtered['Fehler_Rate'] > 0.0]

# Noiseless künstlich für Bitflip und Depolarizing bei p=0.0 hinzufügen
base_bitflip = pd.DataFrame([{'Fehler_Rate': 0.0, 'Fehler_Art': 'bitflip', 'Berechnete_Energie_Hartree': noiseless_energy}])
base_depol = pd.DataFrame([{'Fehler_Rate': 0.0, 'Fehler_Art': 'depolarizing', 'Berechnete_Energie_Hartree': noiseless_energy}])

df_plot = pd.concat([base_bitflip, base_depol, df_noisy[['Fehler_Rate', 'Fehler_Art', 'Berechnete_Energie_Hartree']]], ignore_index=True)

# 4. Plot-Design konfigurieren (Seaborn)
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Graphen zeichnen
sns.lineplot(data=df_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Fehler_Art', style='Fehler_Art', markers=True, dashes=False, 
             palette=['#e74c3c', '#3498db'], linewidth=2)

# Exakte Energie und Chemical Accuracy einzeichnen
chem_acc = 0.0016
plt.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5, label='Exakte Grundzustandsenergie')
plt.fill_between(x=[-0.005, 0.105], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                 color='green', alpha=0.2, label='Chemical Accuracy ($\pm 1.6 \cdot 10^{-3}$ Ha)')

# Achsen und Layout
plt.xlim(-0.002, 0.102)
plt.title('Einfluss der Fehlerintensität (p) auf die VQE-Energie (H$_2$, TwoLocal, Tiefe 1, COBYLA)', fontsize=14)
plt.xlabel('Fehlerwahrscheinlichkeit $p$', fontsize=12)
plt.ylabel('Berechnete Energie (Hartree)', fontsize=12)

# Legende aufhübschen
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Bitflip', 'Depolarizing', 'Exakte Grundzustandsenergie', 'Chemical Accuracy'], 
           loc='upper left', fontsize=11, framealpha=0.9)

# --- Inset: Vergrößerte Ansicht der ersten 10 Datenpunkte ---
ax = plt.gca()

# Ersten 10 x-Werte bestimmen
first_10_p = sorted(df_plot['Fehler_Rate'].unique())[:10]
max_p_inset = first_10_p[-1]

# Inset in der unteren rechten Ecke platzieren (x0, y0, width, height)
axins = ax.inset_axes([0.55, 0.15, 0.4, 0.4])

# Daten im Inset erneut zeichnen (ohne Legende)
sns.lineplot(data=df_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Fehler_Art', style='Fehler_Art', markers=True, dashes=False, 
             palette=['#e74c3c', '#3498db'], linewidth=2, ax=axins, legend=False)

# Hilfslinien im Inset
axins.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5)
axins.fill_between(x=[-0.001, max_p_inset + 0.001], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                   color='green', alpha=0.2)

# Limits für das Inset dynamisch berechnen
inset_df = df_plot[df_plot['Fehler_Rate'] <= max_p_inset]
min_y_inset = min(exact_energy - 0.002, inset_df['Berechnete_Energie_Hartree'].min() - 0.001)
max_y_inset = max(exact_energy + chem_acc + 0.001, inset_df['Berechnete_Energie_Hartree'].max() + 0.002)

axins.set_xlim(-0.0002, max_p_inset + 0.0002)
axins.set_ylim(min_y_inset, max_y_inset)
axins.set_xlabel('')
axins.set_ylabel('')

# Zoom-Indikator zeichnen (Verbindungslinien vom Hauptgraph zum Inset)
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.savefig('images/plot_error_intensity_h2.pdf', format='pdf')
print("Graph erfolgreich als 'images/plot_error_intensity_h2.pdf' gespeichert.")

# =========================================================
# GRAPH 2: Skalierung der Schaltungstiefe (Depth Scaling)
# =========================================================
print("\nErstelle Graph für 'Skalierung der Schaltungstiefe'...")

# 1. Daten filtern (H2, TwoLocal, COBYLA, Depolarizing Noise)
# Wir nehmen nur Depolarizing, da sich hier die Gatteranzahl am stärksten auswirkt
df_depth = df[(df['Molekuel'] == 'H2_2_Qubits') & 
              (df['Ansatz_Art'] == 'TwoLocal') & 
              (df['Optimierer'] == 'COBYLA')].copy()

# Noiseless (p=0) und Noisy (p>0, nur Depolarizing) trennen
df_depth_noiseless = df_depth[df_depth['Fehler_Rate'] == 0.0].copy()
df_depth_noisy = df_depth[df_depth['Fehler_Art'] == 'depolarizing'].copy()

# Beide zusammenfügen (damit die Graphen bei p=0 starten)
df_depth_plot = pd.concat([df_depth_noiseless, df_depth_noisy], ignore_index=True)
df_depth_plot['Ansatz_Tiefe_Label'] = 'Tiefe ' + df_depth_plot['Ansatz_Tiefe'].astype(str)
df_depth_plot = df_depth_plot.sort_values(by=['Ansatz_Tiefe', 'Fehler_Rate'])

# 2. Plot initialisieren
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.lineplot(data=df_depth_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Ansatz_Tiefe_Label', style='Ansatz_Tiefe_Label', markers=True, dashes=False, 
             palette='magma', linewidth=2)

plt.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5, label='Exakte Grundzustandsenergie')
plt.fill_between(x=[-0.005, 0.105], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                 color='green', alpha=0.2, label='Chemical Accuracy')

plt.xlim(-0.002, 0.102)
plt.title('Skalierung der Schaltungstiefe unter Depolarizing Noise (H$_2$, TwoLocal, COBYLA)', fontsize=14)
plt.xlabel('Fehlerwahrscheinlichkeit $p$', fontsize=12)
plt.ylabel('Berechnete Energie (Hartree)', fontsize=12)

plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

# --- Inset: Vergrößerte Ansicht der ersten 10 Datenpunkte ---
ax2 = plt.gca()

# Ersten 10 x-Werte bestimmen
first_10_p_depth = sorted(df_depth_plot['Fehler_Rate'].unique())[:10]
max_p_inset_depth = first_10_p_depth[-1]

# Inset in der unteren rechten Ecke platzieren (x0, y0, width, height)
axins2 = ax2.inset_axes([0.55, 0.15, 0.4, 0.4])

# Daten im Inset erneut zeichnen (ohne Legende)
sns.lineplot(data=df_depth_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Ansatz_Tiefe_Label', style='Ansatz_Tiefe_Label', markers=True, dashes=False, 
             palette='magma', linewidth=2, ax=axins2, legend=False)

# Hilfslinien im Inset
axins2.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5)
axins2.fill_between(x=[-0.001, max_p_inset_depth + 0.001], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                   color='green', alpha=0.2)

# Limits für das Inset dynamisch berechnen
inset_df_depth = df_depth_plot[df_depth_plot['Fehler_Rate'] <= max_p_inset_depth]
min_y_inset_depth = min(exact_energy - 0.002, inset_df_depth['Berechnete_Energie_Hartree'].min() - 0.001)
max_y_inset_depth = max(exact_energy + chem_acc + 0.001, inset_df_depth['Berechnete_Energie_Hartree'].max() + 0.002)

axins2.set_xlim(-0.0002, max_p_inset_depth + 0.0002)
axins2.set_ylim(min_y_inset_depth, max_y_inset_depth)
axins2.set_xlabel('')
axins2.set_ylabel('')

# Zoom-Indikator zeichnen (Verbindungslinien vom Hauptgraph zum Inset)
ax2.indicate_inset_zoom(axins2, edgecolor="black")

plt.tight_layout()
plt.savefig('images/plot_depth_scaling_h2.pdf', format='pdf')
print("Graph erfolgreich als 'images/plot_depth_scaling_h2.pdf' gespeichert.")

# =========================================================
# GRAPH 3: Vergleich der Optimierungsverfahren (COBYLA vs. SPSA)
# =========================================================
print("\nErstelle Graph für 'Vergleich der Optimierungsverfahren'...")

# 1. Daten filtern (H2, TwoLocal, Tiefe 1, Depolarizing Noise)
df_opt = df[(df['Molekuel'] == 'H2_2_Qubits') & 
            (df['Ansatz_Tiefe'] == 1) & 
            (df['Ansatz_Art'] == 'TwoLocal')].copy()

# Noiseless und Depolarizing isolieren und zusammenfügen
df_opt_noiseless = df_opt[df_opt['Fehler_Rate'] == 0.0].copy()
df_opt_noisy = df_opt[df_opt['Fehler_Art'] == 'depolarizing'].copy()

df_opt_plot = pd.concat([df_opt_noiseless, df_opt_noisy], ignore_index=True)
df_opt_plot = df_opt_plot.sort_values(by=['Optimierer', 'Fehler_Rate'])

# 2. Plot initialisieren
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.lineplot(data=df_opt_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Optimierer', style='Optimierer', markers=True, dashes=False, 
             palette=['#2980b9', '#f39c12'], linewidth=2) # Blau (COBYLA), Orange (SPSA)

plt.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5, label='Exakte Grundzustandsenergie')
plt.fill_between(x=[-0.005, 0.105], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                 color='green', alpha=0.2, label='Chemical Accuracy')

plt.xlim(-0.002, 0.102)
plt.title('Vergleich der Optimierer unter Depolarizing Noise (H$_2$, TwoLocal, Tiefe 1)', fontsize=14)
plt.xlabel('Fehlerwahrscheinlichkeit $p$', fontsize=12)
plt.ylabel('Berechnete Energie (Hartree)', fontsize=12)

plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

# --- Inset: Vergrößerte Ansicht der ersten 10 Datenpunkte ---
ax3 = plt.gca()
first_10_p_opt = sorted(df_opt_plot['Fehler_Rate'].unique())[:10]
max_p_inset_opt = first_10_p_opt[-1]

axins3 = ax3.inset_axes([0.55, 0.15, 0.4, 0.4])
sns.lineplot(data=df_opt_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
             hue='Optimierer', style='Optimierer', markers=True, dashes=False, 
             palette=['#2980b9', '#f39c12'], linewidth=2, ax=axins3, legend=False)

axins3.axhline(exact_energy, color='black', linestyle='--', linewidth=1.5)
axins3.fill_between(x=[-0.001, max_p_inset_opt + 0.001], y1=exact_energy - chem_acc, y2=exact_energy + chem_acc, 
                   color='green', alpha=0.2)

inset_df_opt = df_opt_plot[df_opt_plot['Fehler_Rate'] <= max_p_inset_opt]
min_y_inset_opt = min(exact_energy - 0.002, inset_df_opt['Berechnete_Energie_Hartree'].min() - 0.001)
max_y_inset_opt = max(exact_energy + chem_acc + 0.001, inset_df_opt['Berechnete_Energie_Hartree'].max() + 0.002)

axins3.set_xlim(-0.0002, max_p_inset_opt + 0.0002)
axins3.set_ylim(min_y_inset_opt, max_y_inset_opt)
axins3.set_xlabel('')
axins3.set_ylabel('')

ax3.indicate_inset_zoom(axins3, edgecolor="black")

plt.tight_layout()
plt.savefig('images/plot_optimizer_comparison_h2.pdf', format='pdf')
print("Graph erfolgreich als 'images/plot_optimizer_comparison_h2.pdf' gespeichert.")

# =========================================================
# GRAPH 4: Performance-Profil: COBYLA vs. SPSA (Doppel-Graph)
# =========================================================
print("\nErstelle Graph für 'Performance-Profil: COBYLA vs. SPSA'...")

# Kombinierten Datensatz aus H2 und LiH erstellen
try:
    df_lih = pd.read_csv('sim_data/vqe_masterarbeit_results_LiH8.csv')
    df_all = pd.concat([df, df_lih], ignore_index=True)
except FileNotFoundError:
    print("Hinweis: 'sim_data/vqe_masterarbeit_results_LiH8.csv' nicht gefunden. Verwende nur H2-Datensatz.")
    df_all = df

# Konsistente Farbpalette: COBYLA in Blau, SPSA in Orange
opt_palette = {'COBYLA': '#2980b9', 'SPSA': '#f39c12'}

# 1. Daten für Subplot 1 (Links): Robustheit LiH, Depolarizing, Tiefe 2
df_sub1 = df_all[(df_all['Molekuel'].str.contains('LiH', na=False)) & 
                 (df_all['Ansatz_Tiefe'] == 2) & 
                 (df_all['Ansatz_Art'] == 'TwoLocal') &
                 (df_all['Fehler_Rate'] <= 0.1)].copy()

if not df_sub1.empty:
    exact_energy_lih = df_sub1['Exakte_Energie_Hartree'].iloc[0]
    
    # "noiseless" und "depolarizing" zusammenfügen, damit Linien bei 0 starten
    df_sub1_noiseless = df_sub1[df_sub1['Fehler_Rate'] == 0.0].copy()
    df_sub1_noisy = df_sub1[df_sub1['Fehler_Art'] == 'depolarizing'].copy()
    
    df_sub1_plot = pd.concat([df_sub1_noiseless, df_sub1_noisy], ignore_index=True)
    df_sub1_plot = df_sub1_plot.sort_values(by=['Optimierer', 'Fehler_Rate'])
else:
    df_sub1_plot = pd.DataFrame()
    print("Warnung: Keine LiH-Daten für den linken Subplot gefunden.")

# 2. Daten für Subplot 2 (Rechts): Durchschnittliche Rechenzeit (Moleküle x Optimierer)
df_sub2 = df_all.groupby(['Molekuel', 'Optimierer'])['Dauer_Sekunden'].mean().reset_index()

# 3. Canvas (Figure) anlegen: 1 Zeile, 2 Spalten
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

# ---- Subplot 1 (Links): Linien-Graph (Robustheit) ----
if not df_sub1_plot.empty:
    sns.lineplot(data=df_sub1_plot, x='Fehler_Rate', y='Berechnete_Energie_Hartree', 
                 hue='Optimierer', style='Optimierer', markers=True, dashes=False, 
                 palette=opt_palette, linewidth=2, ax=axes[0])
    
    axes[0].axhline(exact_energy_lih, color='black', linestyle='--', linewidth=1.5, label='Exakte Grundzustandsenergie')
    axes[0].fill_between(x=[-0.005, 0.105], y1=exact_energy_lih - chem_acc, y2=exact_energy_lih + chem_acc, 
                         color='green', alpha=0.2, label='Chemical Accuracy')
    
    axes[0].set_xlim(-0.002, 0.102)
    axes[0].set_title('Robustheit unter Rauschen (LiH, TwoLocal, Tiefe 2, Depolarizing)', fontsize=13)
    axes[0].set_xlabel('Fehlerwahrscheinlichkeit $p$', fontsize=12)
    axes[0].set_ylabel('Berechnete Energie (Hartree)', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=11, framealpha=0.9)
else:
    axes[0].set_title('Robustheit unter Rauschen (LiH) - Keine Daten', fontsize=13)

# ---- Subplot 2 (Rechts): Bar-Chart (Rechenzeit auf logarithmischer Y-Achse) ----
if not df_sub2.empty:
    sns.barplot(data=df_sub2, x='Molekuel', y='Dauer_Sekunden', hue='Optimierer', 
                palette=opt_palette, ax=axes[1])
    
    axes[1].set_yscale('log')
    axes[1].set_title('Der Preis der Stochastik: Rechenzeit pro Lauf (log-scale)', fontsize=13)
    axes[1].set_xlabel('Molekül', fontsize=12)
    axes[1].set_ylabel('Durchschnittliche Rechenzeit (s)', fontsize=12)
    
    # Die Werte als Text auf die Balken drucken
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f s', padding=3, fontsize=10)
        
    axes[1].legend(title='Optimierer', loc='upper left', fontsize=11)
else:
    axes[1].set_title('Rechenzeit - Keine Daten', fontsize=13)

plt.tight_layout()
plt.savefig('images/plot_performance_profile_cobyla_vs_spsa.pdf', format='pdf')
print("Graph erfolgreich als 'images/plot_performance_profile_cobyla_vs_spsa.pdf' gespeichert.")

# =========================================================
# GRAPH 5: Boxplots (Streuung der Ergebnisse unter Rauschen)
# =========================================================
print("\nErstelle Graph für 'Boxplot der Energieabweichung'...")

# Wir nutzen df_all (H2 + LiH kombiniert), das in Graph 4 geladen wurde.
if 'df_all' in locals() and not df_all.empty:
    # 1. Absolute Energieabweichung berechnen
    df_all['Absolute_Energieabweichung'] = (df_all['Berechnete_Energie_Hartree'] - df_all['Exakte_Energie_Hartree']).abs()
    
    # 2. Filtern: Nur Runs mit Rauschen (p > 0), um die Streuung durch Fehler zu bewerten
    df_box = df_all[df_all['Fehler_Rate'] > 0.0].copy()
    
    # Zur besseren Lesbarkeit die Molekül-Namen umbenennen
    df_box['Molekuel'] = df_box['Molekuel'].replace({
        'H2_2_Qubits': 'H2 (2 Qubits)',
        'LiH_Reduced_4_Qubits': 'LiH (4 Qubits)',
        'LiH_Full_12_Qubits': 'LiH (12 Qubits)'
    })
    
    # 3. Boxplot erstellen
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # X-Achse: Optimierer, Y-Achse: Abweichung, Hue: Molekül
    sns.boxplot(data=df_box, x='Optimierer', y='Absolute_Energieabweichung', hue='Molekuel', palette='Set2')
    
    # Y-Achse auf logarithmische Skala setzen, da Ausreißer sonst den Boxplot stauchen
    plt.yscale('log')
    
    # Chemical Accuracy als Referenz einzeichnen
    plt.axhline(chem_acc, color='green', linestyle='--', linewidth=1.5, label='Chemical Accuracy')
    
    plt.title('Verteilung der absoluten Energieabweichung unter Rauschen ($p > 0$)', fontsize=14)
    plt.xlabel('Optimierer', fontsize=12)
    plt.ylabel('Absolute Energieabweichung (Hartree, log-scale)', fontsize=12)
    
    plt.legend(title='Molekül', loc='lower right', fontsize=11)
    plt.tight_layout()
    
    plt.savefig('images/plot_boxplot_energy_deviation.pdf', format='pdf')
    print("Graph erfolgreich als 'images/plot_boxplot_energy_deviation.pdf' gespeichert.")
else:
    print("Konnte Boxplot nicht erstellen: 'df_all' nicht gefunden oder leer.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten laden
df = pd.read_csv('sim_data/vqe_masterarbeit_results_LiH8.csv')

# Wir filtern auf ein Basis-Szenario für den direkten Vergleich (H2, TwoLocal, Tiefe 4)
df_base = df[(df['Ansatz_Art'] == 'TwoLocal') & (df['Ansatz_Tiefe'] == 4)].copy()
exact_energy = df_base['Exakte_Energie_Hartree'].iloc[0]

# 2. Plot-Umgebung aufbauen (1 Zeile, 2 Spalten)
# Breiten-Verhältnis anpassen (Linienplot braucht mehr Platz)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 2.2]})
sns.set_theme(style="whitegrid")

# Farben definieren (Blau für COBYLA, Rot/Orange für SPSA)
palette = {'COBYLA': '#3498db', 'SPSA': '#e74c3c'}

# --- GRAPH 1: Boxplot der Optimierer-Schritte (Early Stopping beweisen) ---
# Boxplot leicht transparent machen und Ausreißer entfernen (werden vom Stripplot gezeichnet)
sns.boxplot(data=df_base, x='Optimierer', y='Optimierer_Schritte', 
            palette=palette, ax=ax1, width=0.4, linewidth=2, fliersize=0, boxprops={'alpha': 0.4})
# Stripplot zeigt die wirkliche Datendichte (SPSA ist ein dichter Block bei 500)
sns.stripplot(data=df_base, x='Optimierer', y='Optimierer_Schritte', 
              palette=palette, ax=ax1, size=4, jitter=True, alpha=0.7)

ax1.set_title('Verhalten der Optimierer (Schaltungsevaluierungen)\nLiH$_8$, TwoLocal, Tiefe 4', fontsize=13, pad=10)
ax1.set_xlabel('Optimierungsverfahren', fontsize=12)
ax1.set_ylabel('Anzahl der Auswertungen', fontsize=12)
ax1.set_ylim(0, 550)

# --- GRAPH 2: Linienplot der Energie unter Depolarizing Noise (Resilienz beweisen) ---
# Nur Rauschen filtern
df_depol = df_base[df_base['Fehler_Art'] == 'depolarizing'].copy()

# Abweichung zum Idealwert berechnen
df_depol['Energieabweichung'] = df_depol['Berechnete_Energie_Hartree'] - exact_energy

# markers=False ist extrem wichtig, damit die 100 Punkte nicht ineinander verschmieren!
sns.lineplot(data=df_depol, x='Fehler_Rate', y='Energieabweichung', 
             hue='Optimierer', palette=palette, linewidth=2.5, alpha=0.9, ax=ax2)

# Exakte Energie als Referenz einzeichnen
ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Exakte Grundzustandsenergie (Referenz)')

# Chemical Accuracy hinzufügen
chem_acc = 0.0016
ax2.fill_between(x=[-0.005, 0.105], y1=-chem_acc, y2=chem_acc, 
                 color='green', alpha=0.2, label='Chemical Accuracy')

ax2.set_title('Resilienz unter Depolarisierungsrauschen\nLiH$_8$, TwoLocal, Tiefe 4', fontsize=13, pad=10)
ax2.set_xlabel('Fehlerwahrscheinlichkeit $p$', fontsize=12)
ax2.set_ylabel('Energieabweichung (Hartree)', fontsize=12)
ax2.set_xlim(-0.002, 0.102)

# Legende aufräumen
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels, loc='upper left', fontsize=11, framealpha=0.9)

# Layout anpassen und speichern
plt.tight_layout()
plt.savefig('images/plot_optimizer_comparison.pdf', format='pdf', bbox_inches='tight')
print("Graph erfolgreich als 'images/plot_optimizer_comparison.pdf' gespeichert.")
