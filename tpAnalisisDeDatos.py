import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración general
sns.set_style("whitegrid")
os.makedirs('graficos', exist_ok=True)

nombres_aglo = {32: 'CABA', 33: 'GBA'}
colores_aglo = {32: '#1f77b4', 33: '#ff7f0e'}  # azul / naranja, consistente en todos los gráficos


# =========================================================
# 1. CARGA Y UNIFICACIÓN DE LAS BASES EPH (2016-2025)
# =========================================================
ruta_carpeta = 'datos_eph/*.txt'
archivos = glob.glob(ruta_carpeta)

print(f"Se encontraron {len(archivos)} bases de la EPH para unir. Empezando la lectura... (puede tardar unos minutos)\n")

columnas_necesarias = [
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'PONDERA', 'PONDIIO',
    'CH04', 'CH06', 'NIVEL_ED', 'ESTADO', 'CAT_OCUP',
    'PP04B_COD', 'PP04D_COD', 'P21', 'P47T', 'PP3E_TOT', 'PP07H', 'PP07A'
]

codigos_aglomerados = [32, 33]  # 32 = CABA, 33 = Partidos del GBA

lista_dataframes = []
for archivo in archivos:
    try:
        df_temporal = pd.read_csv(archivo, sep=';', usecols=lambda c: c in columnas_necesarias, low_memory=False)
        df_filtrado = df_temporal[df_temporal['AGLOMERADO'].isin(codigos_aglomerados)]
        lista_dataframes.append(df_filtrado)
    except Exception as e:
        print(f"Error leyendo el archivo {archivo}: {e}")

df_historico = pd.concat(lista_dataframes, ignore_index=True)
print(f"Base histórica (2016-2025) unificada: {df_historico.shape[0]:,} registros y {df_historico.shape[1]} columnas (CABA + GBA).\n")


# =========================================================
# 2. OBJETIVO 1.A - NO RESPUESTA AL INGRESO (P21) /////////
# =========================================================
print("=" * 60)
print("OBJETIVO 1.A - NO RESPUESTA A INGRESOS (P21)")
print("=" * 60)

# Nos quedamos con la población ocupada (ESTADO == 1)
df_ocupados = df_historico[df_historico['ESTADO'] == 1].copy()
df_ocupados['Sin_Respuesta'] = (df_ocupados['P21'] == -9).astype(int)

# Resumen global del período completo
total_ocupados = len(df_ocupados)
total_sin_respuesta = df_ocupados['Sin_Respuesta'].sum()
print(f"Total de ocupados (2016-2025, CABA+GBA): {total_ocupados:,}")
print(f"No declaran ingreso de la ocupación principal (-9): {total_sin_respuesta:,} "
      f"({total_sin_respuesta / total_ocupados * 100:.2f}%)\n")

# Evolución por año y aglomerado (para el gráfico)
no_respuesta_anual = df_ocupados.groupby(['ANO4', 'AGLOMERADO']).agg(
    Total_Ocupados=('Sin_Respuesta', 'size'),
    Sin_Respuesta=('Sin_Respuesta', 'sum')
).reset_index()
no_respuesta_anual['Tasa_NR (%)'] = (no_respuesta_anual['Sin_Respuesta'] / no_respuesta_anual['Total_Ocupados'] * 100).round(1)
no_respuesta_anual['Aglomerado'] = no_respuesta_anual['AGLOMERADO'].map(nombres_aglo)

print(no_respuesta_anual[['ANO4', 'Aglomerado', 'Total_Ocupados', 'Sin_Respuesta', 'Tasa_NR (%)']].to_string(index=False))

# --- Gráfico: evolución de la tasa de no respuesta ---
fig, ax = plt.subplots(figsize=(9, 5))
for aglo_cod, aglo_nombre in nombres_aglo.items():
    datos = no_respuesta_anual[no_respuesta_anual['AGLOMERADO'] == aglo_cod].sort_values('ANO4')
    ax.plot(datos['ANO4'], datos['Tasa_NR (%)'], marker='o', label=aglo_nombre, color=colores_aglo[aglo_cod])

ax.set_title('Evolución de la Tasa de No Respuesta al Ingreso (P21)\nPoblación Ocupada, CABA vs GBA (2016-2025)')
ax.set_xlabel('Año')
ax.set_ylabel('Tasa de No Respuesta (%)')
ax.legend(title='Aglomerado')
ax.set_xticks(range(2016, 2026))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/01_no_respuesta_evolucion.png', dpi=150)
plt.show()

# Reemplazamos los -9 por NaN para los cálculos numéricos posteriores
df_ocupados['P21'] = df_ocupados['P21'].replace(-9, np.nan)


# =========================================================
# 3. OBJETIVO 1.B - EXPLORACIÓN UNIVARIADA DEL INGRESO (P21)
#    Y DETECCIÓN DE OUTLIERS
# =========================================================
#
# NOTA METODOLÓGICA:
# P21 está en pesos CORRIENTES. Si calculamos los cuartiles/IQR
# mezclando 2016 a 2025, los valores nominales de los últimos años
# (varias veces más altos solo por inflación) van a inflar
# artificialmente el límite de outliers y distorsionar todo el análisis.
#
# Por eso, para esta exploración univariada usamos un CORTE TRANSVERSAL
# (el último trimestre disponible de la base). La comparación histórica
# !de ingresos en términos REALES (ajustados por IPC) la vamos a hacer
# !más adelante, en la sección de evolución de ingresos en el objetivo 2.

ultimo_ano = df_ocupados['ANO4'].max()
ultimo_trim = df_ocupados.loc[df_ocupados['ANO4'] == ultimo_ano, 'TRIMESTRE'].max()

print("\n" + "=" * 60)
print(f"OBJETIVO 1.B - DISTRIBUCIÓN DEL INGRESO (P21)")
print(f"Corte transversal: {ultimo_ano} - Trimestre {ultimo_trim}")
print("=" * 60)

df_reciente = df_ocupados[(df_ocupados['ANO4'] == ultimo_ano) & (df_ocupados['TRIMESTRE'] == ultimo_trim)]

estadisticos = {}
outliers_info = {}

for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()

    Q1 = ingresos.quantile(0.25)
    Q2 = ingresos.quantile(0.50)
    Q3 = ingresos.quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    outliers = ingresos[ingresos > limite_superior]

    estadisticos[aglo_cod] = {
        'n': len(ingresos),
        'media': ingresos.mean(),
        'mediana': Q2,
        'std': ingresos.std(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'min': ingresos.min(),
        'max': ingresos.max(),
    }
    outliers_info[aglo_cod] = {
        'limite_superior': limite_superior,
        'cantidad': len(outliers),
        'porcentaje': len(outliers) / len(ingresos) * 100,
    }

    print(f"\n{aglo_nombre} (n={len(ingresos)}):")
    print(f"  - Media:   ${ingresos.mean():>14,.0f}")
    print(f"  - Mediana: ${Q2:>14,.0f}")
    print(f"  - Q1 / Q3: ${Q1:>14,.0f}  /  ${Q3:>14,.0f}")
    print(f"  - IQR:     ${IQR:>14,.0f}")
    print(f"  - Límite atípico (Q3 + 1.5*IQR): ${limite_superior:,.0f}")
    print(f"  - Outliers detectados: {len(outliers)} casos ({outliers_info[aglo_cod]['porcentaje']:.2f}%)")

print("-" * 60)

# Tabla resumen (útil para pegar directo en el informe)
tabla_resumen = pd.DataFrame(estadisticos).T
tabla_resumen.index = tabla_resumen.index.map(nombres_aglo)
tabla_resumen = tabla_resumen.round(0)
print("\nTabla resumen - Estadísticos descriptivos de P21 (corte más reciente):")
print(tabla_resumen.to_string())


# --- Gráfico 1: Boxplot comparativo con umbral de outliers ---
fig, ax = plt.subplots(figsize=(8, 6))

data_to_plot = []
tick_labels_bp = []
for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    data_to_plot.append(ingresos)
    tick_labels_bp.append(aglo_nombre)

bp = ax.boxplot(data_to_plot, tick_labels=tick_labels_bp, patch_artist=True, showfliers=False)

for patch, aglo_cod in zip(bp['boxes'], nombres_aglo.keys()):
    patch.set_facecolor(colores_aglo[aglo_cod])
    patch.set_alpha(0.6)

# Calculamos el Y máximo a mostrar: 1.3× el límite más alto (CABA = 3.8M → ~5M)
y_max = max(outliers_info[c]['limite_superior'] for c in nombres_aglo) * 1.3
ax.set_ylim(0, y_max)

# Outliers visibles: solo los que quedan dentro del rango graficado
for i, aglo_cod in enumerate(nombres_aglo.keys(), start=1):
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    limite = outliers_info[aglo_cod]['limite_superior']
    outliers_visibles = ingresos[(ingresos > limite) & (ingresos <= y_max)]
    ax.scatter([i] * len(outliers_visibles), outliers_visibles,
               color=colores_aglo[aglo_cod], alpha=0.35, s=18, zorder=3)

# Anotaciones: mediana y línea de límite outlier
offsets = {32: (0.30, -0.30), 33: (0.30, -0.30)}  # ajuste horizontal del texto
for i, aglo_cod in enumerate(nombres_aglo.keys(), start=1):
    est = estadisticos[aglo_cod]
    out = outliers_info[aglo_cod]
    # Mediana
    ax.text(i + 0.28, est['mediana'],
            f"Mediana:\n${est['mediana']/1_000_000:.2f}M",
            fontsize=8, va='center', color=colores_aglo[aglo_cod])
    # Línea y etiqueta del límite outlier
    ax.axhline(out['limite_superior'], color=colores_aglo[aglo_cod],
               linestyle='--', alpha=0.7, linewidth=1.2)
    ax.text(i + 0.28, out['limite_superior'],
            f"Límite outlier:\n${out['limite_superior']/1_000_000:.2f}M",
            fontsize=8, va='bottom', color=colores_aglo[aglo_cod])

# Eje Y en millones de pesos
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M'))
ax.set_title(f"Análisis de Posición: Cuartiles, Mediana y Umbral de Valores Atípicos\nCABA vs GBA — {ultimo_ano} T{ultimo_trim}")
ax.set_ylabel("Ingreso de la Ocupación Principal ($ millones corrientes)")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/02_boxplot_ingresos.png', dpi=150)
plt.show()

# --- Gráfico 2: Histograma + curva de densidad (KDE), sin outliers ---
fig, ax = plt.subplots(figsize=(9, 5))

for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    limite_superior = outliers_info[aglo_cod]['limite_superior']
    ingresos_plot = ingresos[ingresos <= limite_superior]  # recorte visual para legibilidad

    sns.histplot(ingresos_plot, stat='density', kde=True, label=aglo_nombre,
                  color=colores_aglo[aglo_cod], alpha=0.4, ax=ax)
    ax.axvline(estadisticos[aglo_cod]['mediana'], color=colores_aglo[aglo_cod], linestyle='--',
               label=f"Mediana {aglo_nombre}: ${estadisticos[aglo_cod]['mediana']:,.0f}")

ax.set_title(f"Distribución del Ingreso de la Ocupación Principal (P21)\nCABA vs GBA - {ultimo_ano} T{ultimo_trim} (sin outliers)")
# ax.set_ylim(0, 8_000_000)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M'))
ax.set_xlabel("Ingreso de la Ocupación Principal ($)")
ax.set_ylabel("Densidad")
ax.legend()
plt.tight_layout()
plt.savefig('graficos/03_histograma_ingresos.png', dpi=150)
plt.show()

print("\nGráficos guardados en la carpeta 'graficos/':")
print("  - 01_no_respuesta_evolucion.png")
print("  - 02_boxplot_ingresos.png")
print("  - 03_histograma_ingresos.png")