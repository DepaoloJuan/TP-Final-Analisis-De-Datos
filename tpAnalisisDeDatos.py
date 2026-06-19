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


# =========================================================
# OBJETIVO 2 - ANÁLISIS MULTIVARIADO
# =========================================================

# IPC anual promedio (base 2016 = 100)
# Fuente: INDEC - Índice de Precios al Consumidor
ipc_anual = {
    2016: 100.0,
    2017: 125.5,
    2018: 172.0,
    2019: 269.0,
    2020: 336.0,
    2021: 418.0,
    2022: 628.0,
    2023: 1415.0,
    2024: 4070.0,
    2025: 6850.0
}

df_historico['IPC'] = df_historico['ANO4'].map(ipc_anual)
df_historico['P21_real'] = (df_historico['P21'] / df_historico['IPC']) * 100
print("-" * 60)
print("\nColumnas nuevas agregadas: IPC, P21_real")
print(df_historico[['ANO4', 'P21', 'IPC', 'P21_real']].dropna().head(10))


# =========================================================
# 2.1 - EVOLUCIÓN DEL INGRESO REAL POR SEXO
# =========================================================

# CH04: 1 = Varón, 2 = Mujer
df_ocupados_real = df_historico[
    (df_historico['ESTADO'] == 1) &
    (df_historico['P21'] > 0) &
    (df_historico['P21_real'].notna())
].copy()

df_ocupados_real['Sexo'] = df_ocupados_real['CH04'].map({1: 'Varón', 2: 'Mujer'})

ingreso_sexo = df_ocupados_real.groupby(['ANO4', 'AGLOMERADO', 'Sexo']).agg(
    Mediana_real=('P21_real', 'median')
).reset_index()

ingreso_sexo['Aglomerado'] = ingreso_sexo['AGLOMERADO'].map(nombres_aglo)

# print(ingreso_sexo.to_string(index=False))

# --- Gráfico ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = ingreso_sexo[ingreso_sexo['AGLOMERADO'] == aglo_cod]
    for sexo, linestyle in zip(['Varón', 'Mujer'], ['-', '--']):
        subset = datos[datos['Sexo'] == sexo].sort_values('ANO4')
        ax.plot(subset['ANO4'], subset['Mediana_real'],
                marker='o', linestyle=linestyle, label=sexo)
    ax.set_title(f'{aglo_nombre} — Ingreso Real Mediano por Sexo')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real ($ de 2016)')
    ax.set_xticks(range(2016, 2026))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Sexo')
    ax.grid(alpha=0.3)

plt.suptitle('Evolución del Ingreso Real de la Ocupación Principal por Sexo\nCABA vs GBA (2016-2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/04_ingreso_real_sexo.png', dpi=150)
plt.show()


# =========================================================
# 2.2 - EVOLUCIÓN DEL INGRESO REAL POR NIVEL EDUCATIVO
# =========================================================

# NIVEL_ED:
# 1 = Primaria incompleta
# 2 = Primaria completa
# 3 = Secundaria incompleta
# 4 = Secundaria completa
# 5 = Superior universitaria incompleta
# 6 = Superior universitaria completa
# 7 = Sin instrucción
# 9 = NS/NR

niveles_ed = {
    1: 'Primaria incompleta',
    2: 'Primaria completa',
    3: 'Secundaria incompleta',
    4: 'Secundaria completa',
    5: 'Superior incompleta',
    6: 'Superior completa',
    7: 'Sin instrucción'
}

df_ocupados_real['Nivel_Ed'] = df_ocupados_real['NIVEL_ED'].map(niveles_ed)

# Agrupamos en 3 categorías para que el gráfico sea legible
df_ocupados_real['Nivel_Ed_agrup'] = df_ocupados_real['NIVEL_ED'].map({
    1: 'Bajo (hasta primaria)',
    2: 'Bajo (hasta primaria)',
    7: 'Bajo (hasta primaria)',
    3: 'Medio (secundaria)',
    4: 'Medio (secundaria)',
    5: 'Alto (superior)',
    6: 'Alto (superior)',
})

ingreso_ed = df_ocupados_real.groupby(['ANO4', 'AGLOMERADO', 'Nivel_Ed_agrup']).agg(
    Mediana_real=('P21_real', 'median')
).reset_index()

ingreso_ed = ingreso_ed.dropna(subset=['Nivel_Ed_agrup'])

# --- Gráfico ---
colores_ed = {
    'Bajo (hasta primaria)': '#d62728',
    'Medio (secundaria)':    '#ff7f0e',
    'Alto (superior)':       '#2ca02c'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = ingreso_ed[ingreso_ed['AGLOMERADO'] == aglo_cod]
    for nivel, color in colores_ed.items():
        subset = datos[datos['Nivel_Ed_agrup'] == nivel].sort_values('ANO4')
        ax.plot(subset['ANO4'], subset['Mediana_real'],
                marker='o', label=nivel, color=color)
    ax.set_title(f'{aglo_nombre} — Ingreso Real por Nivel Educativo')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real ($ de 2016)')
    ax.set_xticks(range(2016, 2026))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Nivel educativo', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Evolución del Ingreso Real por Nivel Educativo\nCABA vs GBA (2016-2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/05_ingreso_real_nivel_ed.png', dpi=150)
plt.show()


# =========================================================
# 2.3 - EVOLUCIÓN DEL INGRESO REAL POR CATEGORÍA OCUPACIONAL
# =========================================================

cat_ocup = {
    1: 'Patrón/Empleador',
    2: 'Cuenta propia',
    3: 'Obrero/Empleado',
}

df_ocupados_real['Cat_Ocup'] = df_ocupados_real['CAT_OCUP'].map(cat_ocup)

ingreso_cat = df_ocupados_real.groupby(['ANO4', 'AGLOMERADO', 'Cat_Ocup']).agg(
    Mediana_real=('P21_real', 'median')
).reset_index()

ingreso_cat = ingreso_cat.dropna(subset=['Cat_Ocup'])

# --- Gráfico ---
colores_cat = {
    'Patrón/Empleador': '#9467bd',
    'Cuenta propia':    '#8c564b',
    'Obrero/Empleado':  '#1f77b4',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = ingreso_cat[ingreso_cat['AGLOMERADO'] == aglo_cod]
    for cat, color in colores_cat.items():
        subset = datos[datos['Cat_Ocup'] == cat].sort_values('ANO4')
        if len(subset) > 0:
            ax.plot(subset['ANO4'], subset['Mediana_real'],
                    marker='o', label=cat, color=color)
    ax.set_title(f'{aglo_nombre} — Ingreso Real por Categoría Ocupacional')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real ($ de 2016)')
    ax.set_xticks(range(2016, 2026))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Categoría', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Evolución del Ingreso Real por Categoría Ocupacional\nCABA vs GBA (2016-2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/06_ingreso_real_cat_ocup.png', dpi=150)
plt.show()


# =========================================================
# 2.4 - TASAS LABORALES POR SEXO
# =========================================================

df_activos = df_historico[df_historico['ESTADO'].isin([1, 2, 3])].copy()
df_activos['Sexo'] = df_activos['CH04'].map({1: 'Varón', 2: 'Mujer'})

def calcular_tasas(df):
    ocupados  = df[df['ESTADO'] == 1]['PONDERA'].sum()
    desocupados = df[df['ESTADO'] == 2]['PONDERA'].sum()
    inactivos = df[df['ESTADO'] == 3]['PONDERA'].sum()
    total = ocupados + desocupados + inactivos
    pea = ocupados + desocupados
    return pd.Series({
        'Tasa_Actividad':     pea / total * 100,
        'Tasa_Empleo':        ocupados / total * 100,
        'Tasa_Desocupacion':  desocupados / pea * 100 if pea > 0 else 0
    })

tasas_sexo = df_activos.groupby(['ANO4', 'AGLOMERADO', 'Sexo']).apply(calcular_tasas).reset_index()
tasas_sexo['Aglomerado'] = tasas_sexo['AGLOMERADO'].map(nombres_aglo)

# print(tasas_sexo.to_string(index=False))

# --- Gráfico ---
indicadores = {
    'Tasa_Actividad': 'Tasa de Actividad (%)',
    'Tasa_Empleo': 'Tasa de Empleo (%)',
    'Tasa_Desocupacion': 'Tasa de Desocupación (%)'
}

for indicador, titulo_eje in indicadores.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
        datos = tasas_sexo[tasas_sexo['AGLOMERADO'] == aglo_cod]
        for sexo, linestyle in zip(['Varón', 'Mujer'], ['-', '--']):
            subset = datos[datos['Sexo'] == sexo].sort_values('ANO4')
            ax.plot(subset['ANO4'], subset[indicador],
                    marker='o', linestyle=linestyle, label=sexo)
        ax.set_title(f'{aglo_nombre}')
        ax.set_xlabel('Año')
        ax.set_ylabel(titulo_eje)
        ax.set_xticks(range(2016, 2026))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Sexo')
        ax.grid(alpha=0.3)

    nombre_archivo = indicador.lower()
    plt.suptitle(f'{titulo_eje} por Sexo — CABA vs GBA (2016-2025)', y=0.98)
    plt.tight_layout()
    plt.savefig(f'graficos/07_{nombre_archivo}_sexo.png', dpi=150)
    plt.show()

# =========================================================
# 2.5 - TASAS LABORALES POR GRUPO ETARIO
# =========================================================

# Creamos grupos de edad a partir de CH06
bins = [14, 24, 34, 49, 64, 99]
labels = ['15-24', '25-34', '35-49', '50-64', '65+']

df_activos['Grupo_Edad'] = pd.cut(df_activos['CH06'], bins=bins, labels=labels)

tasas_edad = df_activos.groupby(['ANO4', 'AGLOMERADO', 'Grupo_Edad']).apply(calcular_tasas).reset_index()
tasas_edad = tasas_edad.dropna(subset=['Grupo_Edad'])

# --- Gráfico: solo desocupación por edad (la más informativa) ---
colores_edad = {
    '15-24': '#d62728',
    '25-34': '#ff7f0e',
    '35-49': '#2ca02c',
    '50-64': '#1f77b4',
    '65+':   '#9467bd'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = tasas_edad[tasas_edad['AGLOMERADO'] == aglo_cod]
    for grupo, color in colores_edad.items():
        subset = datos[datos['Grupo_Edad'] == grupo].sort_values('ANO4')
        ax.plot(subset['ANO4'], subset['Tasa_Desocupacion'],
                marker='o', label=grupo, color=color)
    ax.set_title(f'{aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Tasa de Desocupación (%)')
    ax.set_xticks(range(2016, 2026))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Grupo etario')
    ax.grid(alpha=0.3)

plt.suptitle('Tasa de Desocupación por Grupo Etario — CABA vs GBA (2016-2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/08_desocupacion_edad.png', dpi=150)
plt.show()


# --- Gráfico unificado: 3 tasas por sexo ---
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

indicadores = {
    'Tasa_Actividad': 'Tasa de Actividad (%)',
    'Tasa_Empleo': 'Tasa de Empleo (%)',
    'Tasa_Desocupacion': 'Tasa de Desocupación (%)'
}

for row, (indicador, titulo_eje) in enumerate(indicadores.items()):
    for col, (aglo_cod, aglo_nombre) in enumerate(nombres_aglo.items()):
        ax = axes[row, col]
        datos = tasas_sexo[tasas_sexo['AGLOMERADO'] == aglo_cod]
        for sexo, linestyle in zip(['Varón', 'Mujer'], ['-', '--']):
            subset = datos[datos['Sexo'] == sexo].sort_values('ANO4')
            ax.plot(subset['ANO4'], subset[indicador],
                    marker='o', linestyle=linestyle, label=sexo)
        ax.set_title(f'{aglo_nombre} — {titulo_eje}')
        ax.set_xlabel('Año')
        ax.set_ylabel(titulo_eje)
        ax.set_xticks(range(2016, 2026))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Sexo', fontsize=8)
        ax.grid(alpha=0.3)

plt.suptitle('Tasas Laborales por Sexo — CABA vs GBA (2016-2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/07_tasas_laborales_sexo.png', dpi=150)
plt.show()


# =========================================================
# 2.6 - BRECHA SALARIAL DE GÉNERO
# =========================================================

# Pivoteamos para tener varón y mujer en columnas separadas
brecha = ingreso_sexo.pivot_table(
    index=['ANO4', 'AGLOMERADO'],
    columns='Sexo',
    values='Mediana_real'
).reset_index()

brecha.columns.name = None
brecha['Brecha (%)'] = ((brecha['Varón'] - brecha['Mujer']) / brecha['Varón'] * 100).round(1)
brecha['Aglomerado'] = brecha['AGLOMERADO'].map(nombres_aglo)

print("-" * 60)
print(brecha[['ANO4', 'Aglomerado', 'Varón', 'Mujer', 'Brecha (%)']].to_string(index=False))

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(10, 5))

for aglo_cod, aglo_nombre in nombres_aglo.items():
    datos = brecha[brecha['AGLOMERADO'] == aglo_cod].sort_values('ANO4')
    ax.plot(datos['ANO4'], datos['Brecha (%)'],
            marker='o', label=aglo_nombre, color=colores_aglo[aglo_cod])

ax.set_title('Brecha Salarial de Género — CABA vs GBA (2016-2025)', y=0.98)
ax.set_xlabel('Año')
ax.set_ylabel('Brecha salarial (%)')
ax.set_xticks(range(2016, 2026))
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Aglomerado')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/09_brecha_salarial_genero.png', dpi=150)
plt.show()


print("-" * 60)
# print("Valores únicos en PP04B_COD:")
# print(df_historico[df_historico['ESTADO'] == 1]['PP04B_COD'].value_counts().sort_index().head(20))
# print("Total valores únicos PP04B_COD:", df_historico[df_historico['ESTADO'] == 1]['PP04B_COD'].nunique())
# print("\nTop 20 más frecuentes:")
# print(df_historico[df_historico['ESTADO'] == 1]['PP04B_COD'].value_counts().head(20))


# =========================================================
# 2.7 - COMPOSICIÓN DEL EMPLEO POR SECTOR
# =========================================================

def clasificar_sector(cod):
    if pd.isna(cod):
        return None
    cod = int(cod)
    if 100 <= cod <= 399:
        return 'Agro/Minería'
    elif 1000 <= cod <= 3999:
        return 'Industria'
    elif 4100 <= cod <= 4399:
        return 'Construcción'
    elif 4500 <= cod <= 4999 or 5600 <= cod <= 5699:
        return 'Comercio y hotelería'
    elif 4000 <= cod <= 4099:
        return 'Comercio y hotelería'
    elif 6000 <= cod <= 6999:
        return 'Transporte/Finanzas'
    elif 8400 <= cod <= 8499:
        return 'Adm. Pública'
    elif 8500 <= cod <= 8599:
        return 'Educación'
    elif 8600 <= cod <= 8699:
        return 'Salud'
    elif 9700 <= cod <= 9799:
        return 'Serv. Doméstico'
    elif 9000 <= cod <= 9999:
        return 'Otros servicios'
    else:
        return 'Otros'

df_ocupados_real['Sector'] = df_ocupados_real['PP04B_COD'].apply(clasificar_sector)

# Composición por sector, aglomerado y año
composicion = df_ocupados_real.groupby(['ANO4', 'AGLOMERADO', 'Sector'])['PONDERA'].sum().reset_index()
composicion_total = composicion.groupby(['ANO4', 'AGLOMERADO'])['PONDERA'].sum().reset_index()
composicion_total.columns = ['ANO4', 'AGLOMERADO', 'Total']
composicion = composicion.merge(composicion_total, on=['ANO4', 'AGLOMERADO'])
composicion['Participacion (%)'] = (composicion['PONDERA'] / composicion['Total'] * 100).round(1)
composicion = composicion.dropna(subset=['Sector'])

# --- Gráfico: comparación 2016 vs 2025 por sector ---
años_comparar = [2016, 2025]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = composicion[
        (composicion['AGLOMERADO'] == aglo_cod) &
        (composicion['ANO4'].isin(años_comparar))
    ].pivot_table(index='Sector', columns='ANO4', values='Participacion (%)').fillna(0)

    datos.plot(kind='barh', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title(f'{aglo_nombre}')
    ax.set_xlabel('Participación (%)')
    ax.set_ylabel('')
    ax.legend(title='Año')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Composición del Empleo por Sector — CABA vs GBA (2016 vs 2025)', y=0.98)
plt.tight_layout()
plt.savefig('graficos/10_composicion_empleo_sector.png', dpi=150)
plt.show()
