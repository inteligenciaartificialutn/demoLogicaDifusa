import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo Lógica Difusa – Control de Temperatura", layout="centered")
st.title("🔥 Lógica Difusa – Control de Temperatura (Sigmoides)")

st.markdown("""
Este sistema difuso controla un **calefactor inteligente** que ajusta su potencia según:

- El **error térmico** (diferencia entre temperatura deseada y actual)
- La **variación del error** (si la temperatura sube o baja rápido)
""")

# Variables difusas
error = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'error')
delta = ctrl.Antecedent(np.arange(-5, 5, 0.1), 'delta')
potencia = ctrl.Consequent(np.arange(0, 100, 1), 'potencia')

# Funciones de membresía SIGMOIDALES con mucho solapamiento
# Error térmico
error['frío'] = fuzz.zmf(error.universe, -5, 0)   # decreciente
error['ideal'] = fuzz.gaussmf(error.universe, 0, 3)  # campana suave
error['caliente'] = fuzz.smf(error.universe, 0, 5)   # creciente

# Delta de temperatura
delta['bajando'] = fuzz.zmf(delta.universe, -2, 0)
delta['estable'] = fuzz.gaussmf(delta.universe, 0, 1)
delta['subiendo'] = fuzz.smf(delta.universe, 0, 2)

# Potencia
potencia['baja'] = fuzz.zmf(potencia.universe, 25, 50)
potencia['media'] = fuzz.gaussmf(potencia.universe, 50, 20)
potencia['alta'] = fuzz.smf(potencia.universe, 50, 75)

# Reglas difusas
rule1 = ctrl.Rule(error['frío'] & delta['bajando'], potencia['alta'])
rule2 = ctrl.Rule(error['frío'] & delta['estable'], potencia['alta'])
rule3 = ctrl.Rule(error['ideal'] & delta['estable'], potencia['media'])
rule4 = ctrl.Rule(error['caliente'] | delta['subiendo'], potencia['baja'])

sistema_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

# Interfaz Streamlit
temp_deseada = st.slider("Temperatura deseada (°C)", 15, 30, 22)
temp_actual = st.slider("Temperatura actual (°C)", 10, 35, 18)
delta_val = st.slider("Cambio de temperatura (°C/min)", -5.0, 5.0, 0.0, 0.1)

error_val = temp_deseada - temp_actual
sistema.input['error'] = error_val
sistema.input['delta'] = delta_val
sistema.compute()

potencia_val = sistema.output['potencia']

# ---- Gráficos de funciones de membresía en fila ----
st.subheader("Funciones de membresía (sigmoidales) de las variables difusas")
col1, col2, col3 = st.columns(3)

def plot_membership(variable, title):
    fig, ax = plt.subplots(figsize=(3,2))
    for label in variable.terms:
        ax.plot(variable.universe, variable[label].mf, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Grado', fontsize=8)
    ax.set_xlabel(variable.label, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

with col1:
    st.pyplot(plot_membership(error, "Error térmico"))

with col2:
    st.pyplot(plot_membership(delta, "Cambio de temp"))

with col3:
    st.pyplot(plot_membership(potencia, "Potencia"))

# ---- Gráfico de barras del valor output ----
st.subheader("Grado de pertenencia del valor calculado de potencia")
baja_val  = fuzz.interp_membership(potencia.universe, potencia['baja'].mf, potencia_val)
media_val = fuzz.interp_membership(potencia.universe, potencia['media'].mf, potencia_val)
alta_val  = fuzz.interp_membership(potencia.universe, potencia['alta'].mf, potencia_val)

fig, ax = plt.subplots(figsize=(6,3))
ax.bar(['Baja', 'Media', 'Alta'], [baja_val, media_val, alta_val],
       color=['skyblue','orange','green'])
ax.set_ylabel('Grado de pertenencia')
ax.set_title('Salida difusa – Intensidad por categoría')
st.pyplot(fig)