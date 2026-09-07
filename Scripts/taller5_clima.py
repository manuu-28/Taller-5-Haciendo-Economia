# -*- coding: utf-8 -*-
"""
Taller 5 - Doing Economics: Midiendo la temperatura de la Tierra y el CO2
Equipo consultor - [nombres de los 4 integrantes / rol]

Cómo correrlo en Spyder:
  1. Abre tu repo/carpeta TALLER_5 como proyecto (Projects > New Project >
     Existing directory).
  2. Abre este archivo desde Scripts/.
  3. Corre celda por celda con Ctrl+Enter (cada bloque empieza con "# %%").

Estructura de carpetas esperada:
    RawData/  -> datos_taller_5.csv   (temperatura, NASA GISS)
                 co2_data.txt         (CO2, NOAA Mauna Loa, texto crudo)
    Scripts/  -> este archivo
    Output/   -> gráficos y tablas que genera el script
"""

# %% 1. CONFIGURACIÓN ----------------------------------------------------
# Todo lo que puede cambiar (nombres de archivo, mes a analizar) está
# centralizado acá arriba, para no tener que buscar dentro del código.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ARCHIVO_TEMP = os.path.join("RawData", "datos_taller_5.csv")
ARCHIVO_CO2 = os.path.join("RawData", "co2_data.txt")
CARPETA_SALIDA = "Output"
MES = "Jun"          # nombre de columna en el archivo de temperatura
MES_NUM = 6          # número de mes equivalente, para cruzar con el CO2

os.makedirs(os.path.join(CARPETA_SALIDA, "Figuras"), exist_ok=True)
plt.rcParams["figure.dpi"] = 110


# %% 2. FUNCIONES DE APOYO -----------------------------------------------
# Se definen una sola vez y se reutilizan en varias preguntas, así el
# resto del script queda más corto y legible.

def guardar(fig, nombre):
    """Guarda una figura en Output/Figuras/ con un nombre estándar."""
    ruta = os.path.join(CARPETA_SALIDA, "Figuras", nombre)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.show()


def linea_con_cero(serie, titulo, ylabel, color="steelblue"):
    """Gráfico de línea con una referencia horizontal en 0 (el promedio
    del período base 1951-1980)."""
    fig, ax = plt.subplots()
    ax.axhline(0, color="darkorange", linewidth=1)
    serie.plot(ax=ax, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel(ylabel)
    return fig, ax


def tabla_frecuencias(df, año_ini, año_fin, meses=("Jun", "Jul", "Aug")):
    """Tabla de frecuencias de anomalías mensuales en bins de 0.05°C."""
    datos = df.loc[(df.index >= año_ini) & (df.index <= año_fin), list(meses)]
    bins = np.arange(-0.30, 1.10, 0.05)
    return pd.cut(datos.stack(), bins=bins, right=False).value_counts().sort_index()


# %% 3. CARGA DE DATOS ----------------------------------------------------

# --- 3.1 Temperatura (NASA GISS) ----------------------------------------
temp = pd.read_csv(ARCHIVO_TEMP, skiprows=1, na_values="***")
assert "Year" in temp.columns, (
    "No se encontró la columna 'Year'. Abre el CSV y revisa cuántas filas "
    "de encabezado hay que saltar con skiprows."
)
temp = temp.set_index("Year")
print("Temperatura cargada:", temp.shape, "\n", temp.head(), "\n")

# --- 3.2 CO2 (NOAA Mauna Loa) --------------------------------------------
# El archivo descargado del GML Data Finder es texto crudo: ~40 líneas de
# comentario que empiezan con "#", y después datos separados por espacios
# (no comas). Las columnas oficiales de NOAA, en orden, son:
#   year, month, decimal_date, average, deseasonalized, ndays, sdev, unc
co2 = pd.read_csv(
    ARCHIVO_CO2,
    comment="#",
    sep=r"\s+",
    names=["year", "month", "decimal_date", "average",
           "deseasonalized", "ndays", "sdev", "unc"],
    na_values=-99.99,
)
assert co2.shape[1] == 8, (
    f"El archivo trajo {co2.shape[1]} columnas, se esperaban 8. Mándame una "
    "captura de las primeras filas del .txt para ajustar los nombres."
)
print("CO2 cargado:", co2.shape, "\n", co2.head(), "\n")


# %% =====================================================================
# PARTE 1.1 - Anomalías de temperatura
# =========================================================================

# --- 1.1.1: conceptual, sin código --------------------------------------
# Anomalía = diferencia frente al promedio del período base 1951-1980,
# no temperatura absoluta. Se usa porque es comparable entre estaciones
# y a lo largo del tiempo, mientras que la temperatura absoluta varía
# mucho por ubicación y sesga las comparaciones. (Cita el FAQ de GISS.)

# --- 1.1.2 / 1.1.3(i): línea de un mes -----------------------------------
fig, ax = linea_con_cero(
    temp[MES],
    f"Anomalía de temperatura en {MES} - Hemisferio Norte",
    "Anomalía de temperatura (°C)",
)
guardar(fig, f"1_1_2_linea_{MES}.png")

# --- 1.1.3(ii): línea por estación --------------------------------------
fig, ax = plt.subplots()
ax.axhline(0, color="darkorange", linewidth=1)
for est in ["DJF", "MAM", "JJA", "SON"]:
    temp[est].plot(ax=ax, label=est)
ax.set_title("Anomalía de temperatura por estación - Hemisferio Norte")
ax.set_xlabel("Año")
ax.set_ylabel("Anomalía de temperatura (°C)")
ax.legend(title="Estación")
guardar(fig, "1_1_3ii_estaciones.png")

# --- 1.1.3(iii): línea anual ----------------------------------------------
fig, ax = linea_con_cero(
    temp["J-D"],
    "Anomalía de temperatura anual - Hemisferio Norte",
    "Anomalía de temperatura anual (°C)",
    color="firebrick",
)
guardar(fig, "1_1_3iii_anual.png")

# --- 1.1.5 / 1.1.6: discusión, sin código ---------------------------------
# 1.1.5: mes -> mucho ruido interanual; estación -> revela si el
#        calentamiento es parejo entre estaciones; año -> la tendencia de
#        largo plazo más limpia (promedia el ruido mensual).
# 1.1.6: compara la forma de 1.1.3(iii) con la reconstrucción de largo
#        plazo de la Academia Nacional de Ciencias -- el aumento post-1950
#        no tiene precedente en la serie larga.


# %% =====================================================================
# PARTE 1.2 - Variación de la temperatura en el tiempo
# =========================================================================

# --- 1.2.1: tablas de frecuencia -----------------------------------------
frec_51_80 = tabla_frecuencias(temp, 1951, 1980)
frec_81_10 = tabla_frecuencias(temp, 1981, 2010)
print("Frecuencias 1951-1980:\n", frec_51_80)
print("Frecuencias 1981-2010:\n", frec_81_10)

# --- 1.2.2: histogramas comparados ---------------------------------------
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
frec_51_80.plot(kind="bar", ax=axes[0], color="steelblue", title="1951-1980")
frec_81_10.plot(kind="bar", ax=axes[1], color="firebrick", title="1981-2010")
for a in axes:
    a.set_xlabel("Rango de anomalía (°C)")
    a.tick_params(axis="x", rotation=90, labelsize=7)
axes[0].set_ylabel("Frecuencia")
fig.suptitle("Distribución de anomalías de temperatura (JJA)")
guardar(fig, "1_2_2_histogramas.png")
# Discusión: compara moda, dispersión y hacia dónde se corre la masa de
# la distribución entre los dos períodos.

# --- 1.2.3: deciles 3 y 7 (1951-1980) -------------------------------------
valores_51_80 = temp.loc[1951:1980, "Jan":"Dec"].stack()
q30, q70 = np.quantile(valores_51_80, [0.3, 0.7])
print(f"Umbral 'frío' (decil 3): {q30:.3f} °C | Umbral 'caliente' (decil 7): {q70:.3f} °C")

# --- 1.2.4: % de meses "calientes" en 1981-2010 ---------------------------
valores_81_10 = temp.loc[1981:2010, "Jan":"Dec"].stack()
pct_caliente = (valores_81_10 > q70).mean() * 100
print(f"% de meses 'calientes' en 1981-2010 (umbral fijado en 1951-1980): {pct_caliente:.2f}%")

# --- 1.2.5: media y varianza por estación y período -----------------------
largo = (
    temp.loc[:, "DJF":"SON"]
    .stack()
    .reset_index()
    .rename(columns={"level_1": "Estacion", 0: "Valor"})
)
largo["Periodo"] = pd.cut(
    largo["Year"], bins=[1921, 1950, 1980, 2010],
    labels=["1921-1950", "1951-1980", "1981-2010"],
)
media_var = largo.groupby(["Estacion", "Periodo"], observed=True)["Valor"].agg(
    media="mean", varianza="var"
)
print("Media y varianza por estación y período:\n", media_var)
media_var.to_csv(os.path.join(CARPETA_SALIDA, "1_2_5_media_varianza.csv"))

# --- 1.2.6: discusión, sin código -----------------------------------------
# Recomendación de mitigación vs. adaptación, apoyada en 1.2.2, 1.2.4 y
# 1.2.5. Sé explícito sobre qué información adicional necesitarían.


# %% =====================================================================
# PARTE 1.3 - CO2 y su relación con la temperatura
# =========================================================================

# --- 1.3.1 / 1.3.2: discusión, sin código ---------------------------------
# Mauna Loa está lejos de fuentes locales de CO2 y a gran altitud, por lo
# que mide bien la mezcla atmosférica de fondo -> representativo global.
# "average" = promedio mensual crudo; "deseasonalized" (columna "trend"
# de NOAA) le quita el ciclo estacional causado por la fotosíntesis y
# respiración de la vegetación del hemisferio norte.

# --- 1.3.3: línea de CO2 desde 1960 ---------------------------------------
co2_1960 = co2[co2["year"] >= 1960].copy()
co2_1960["fecha"] = pd.to_datetime(dict(year=co2_1960["year"], month=co2_1960["month"], day=1))

fig, ax = plt.subplots()
ax.plot(co2_1960["fecha"], co2_1960["average"], label="Promedio mensual", linewidth=1)
ax.plot(co2_1960["fecha"], co2_1960["deseasonalized"], label="Tendencia (desestacionalizada)", linewidth=1.5)
ax.set_title("Niveles de CO2 - Observatorio de Mauna Loa (1960-presente)")
ax.set_xlabel("Año")
ax.set_ylabel("CO2 (ppm)")
ax.legend()
guardar(fig, "1_3_3_co2_tiempo.png")

# --- 1.3.4: dispersión + correlación de Pearson ---------------------------
co2_mes = co2.loc[co2["month"] == MES_NUM, ["year", "deseasonalized"]].rename(columns={"year": "Year"})
temp_co2 = pd.merge(temp.reset_index()[["Year", MES]], co2_mes, on="Year", how="inner")
assert len(temp_co2) > 0, "El merge no produjo filas; revisa el rango de años en ambos archivos."

fig, ax = plt.subplots()
ax.scatter(temp_co2[MES], temp_co2["deseasonalized"], color="black", s=25)
ax.set_title("Anomalía de temperatura vs. CO2 (tendencia)")
ax.set_xlabel(f"Anomalía de temperatura en {MES} (°C)")
ax.set_ylabel("CO2 - tendencia (ppm)")
guardar(fig, "1_3_4_dispersión.png")

r, p_valor = stats.pearsonr(temp_co2[MES], temp_co2["deseasonalized"])
print(f"Correlación de Pearson ({MES} vs. CO2): r = {r:.3f}  (p = {p_valor:.2e}, n = {len(temp_co2)})")
# Discusión: interpreta magnitud y signo; recuerda que ambas series están
# fuertemente auto-correlacionadas en el tiempo, lo que infla r
# mecánicamente, y que correlación no implica causalidad (ver 1.3.6).

# --- 1.3.6: discusión, sin código -----------------------------------------
# Define correlación espuria vs. causalidad con un ejemplo propio.
# Explica por qué CO2-temperatura, además de la correlación, tiene un
# mecanismo físico plausible (efecto invernadero) que la distingue de
# una correlación puramente espuria.

print("\nListo. Gráficos en Output/Figuras/, tabla en "
      "Output/1_2_5_media_varianza.csv")
