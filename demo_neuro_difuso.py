import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Demo Neuro-Difusa (TF)", layout="wide")
st.title("🧠 Sistema Neuro-Difuso Interactivo con Pesos Visualizados")

# ===================== SLIDERS =====================
st.sidebar.header("Entradas del sistema")
temp_deseada = st.sidebar.slider("Temperatura deseada (°C)", 15, 30, 22)
temp_actual = st.sidebar.slider("Temperatura actual (°C)", 10, 35, 18)
delta_val = st.sidebar.slider("Cambio de temperatura (°C/min)", -5.0, 5.0, 0.0, 0.1)
error_val = temp_deseada - temp_actual

# ===================== DEFUZZIFICATION SELECTOR =====================
metodos = ['centroid', 'bisector', 'mom', 'som', 'lom']
metodo_sel = st.sidebar.selectbox("Método de defuzzificación", metodos, index=0)

def explain_method(method):
    explanations = {
        'centroid': "Centroid: promedio ponderado de todas las áreas bajo la curva de membresía.",
        'bisector': "Bisector: divide el área total en dos partes iguales.",
        'mom': "Mean of Maximum (MOM): promedio de los valores con máxima pertenencia.",
        'som': "Smallest of Maximum (SOM): menor valor con máxima pertenencia.",
        'lom': "Largest of Maximum (LOM): mayor valor con máxima pertenencia."
    }
    return explanations.get(method, "")

st.sidebar.markdown(f"**Descripción método:** {explain_method(metodo_sel)}")

# ===================== DATOS =====================
np.random.seed(42)
n_samples = 300
error_data = np.random.uniform(-10, 10, n_samples)
delta_data = np.random.uniform(-5, 5, n_samples)
potencia_data = np.clip(50 + 3*error_data - 2*delta_data + np.random.normal(0,5,n_samples), 0, 100)

X = np.vstack([error_data, delta_data]).T.astype(np.float32)
y = potencia_data.reshape(-1,1).astype(np.float32)

# Normalización manual
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

# ===================== FUNCIONES =====================
def create_model():
    tf.keras.backend.clear_session()
    model = Sequential([
        Dense(10, activation='tanh', input_shape=(2,)),
        Dense(10, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model):
    model.fit(X_scaled, y_scaled, epochs=50, verbose=0)

def predict(model, error_val, delta_val):
    input_scaled = np.array([[error_val, delta_val]], dtype=np.float32)
    input_scaled = (input_scaled - X_mean) / X_std
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred = pred_scaled * y_std + y_mean
    return pred[0,0]

# ===================== GRAFICOS =====================
def plot_3d(error_val, delta_val, pred):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(error_data, delta_data, potencia_data, c='blue', alpha=0.5, s=20, label='Datos')
    ax.scatter(error_val, delta_val, pred, c='red', s=50, label='Predicción')
    ax.set_xlabel('Error (°C)')
    ax.set_ylabel('Delta (°C/min)')
    ax.set_zlabel('Potencia (%)')
    ax.set_title('Datos + Predicción Neuro-Difusa', fontsize=10)
    ax.legend(fontsize=8)
    return fig

def plot_nn_architecture(model):
    layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    n_layers = len(layers)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis('off')
    x_spacing = 1 / (n_layers + 1)
    positions = []

    # Posiciones de neuronas por capa
    for i, layer in enumerate(layers):
        n_neurons = layer.units
        x = (i+1) * x_spacing
        layer_pos = []
        for j in range(n_neurons):
            y = 1 - (j + 0.5) / n_neurons  # distribución vertical uniforme
            layer_pos.append((x, y))
            circle = plt.Circle((x, y), 0.02, color='skyblue', ec='k')
            ax.add_patch(circle)
            ax.text(x, y+0.03, f"{j+1}", fontsize=6, ha='center')
        positions.append(layer_pos)
        ax.text(x, 1.02, f"{layer.activation.__name__}", fontsize=8, ha='center')
        ax.text(x, -0.05, f"{n_neurons} neuronas", fontsize=6, ha='center')

    # Conexiones con pesos (solo entre capas Dense, sin inputs externos)
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layers[i-1]

        weights = layer.get_weights()[0]  # (n_input, n_output)
        n_input, n_output = weights.shape

        # posiciones de entrada y salida
        x_input = [pos[0] for pos in positions[i-1]]
        y_input = [pos[1] for pos in positions[i-1]]
        x_output = [pos[0] for pos in positions[i]]
        y_output = [pos[1] for pos in positions[i]]

        for j in range(n_input):
            for k in range(n_output):
                x0, y0 = x_input[j], y_input[j]
                x1, y1 = x_output[k], y_output[k]
                weight = weights[j,k]
                lw = max(0.5, abs(weight)*2)
                color = (1,0,0,min(1,abs(weight))) if weight>=0 else (0,0,1,min(1,abs(weight)))
                ax.plot([x0, x1], [y0, y1], color=color, lw=lw)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title("Arquitectura con Pesos (todas conexiones)", fontsize=10)
    return fig

def plot_output_bar(pred):
    # Color progresivo según valor
    if pred < 30:
        color = (0, 0, 1)
    elif pred < 60:
        t = (pred-30)/30
        color = (t, t, 0)
    else:
        t = (pred-60)/40
        t = min(t,1)
        color = (1, 0, 0 + 0.5*(1-t))
    fig, ax = plt.subplots(figsize=(3,2))
    ax.bar(['Potencia'], [pred], color=[color])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Nivel de activación / %')
    ax.set_title('Salida Neuro-Difusa', fontsize=8)
    return fig

# ===================== BOTÓN =====================
if st.sidebar.button("Generar nueva red y reentrenar"):
    model = create_model()
    train_model(model)
else:
    model = create_model()
    train_model(model)

# ===================== PREDICCIÓN =====================
pred = predict(model, error_val, delta_val)
st.metric("Potencia calculada (%)", f"{pred:.1f}")

# ===================== GRAFICOS =====================
col1, col2, col3 = st.columns([1,1,0.6])

with col1:
    st.subheader("Datos + Predicción")
    st.pyplot(plot_3d(error_val, delta_val, pred))

with col2:
    st.subheader("Arquitectura de la Red")
    st.pyplot(plot_nn_architecture(model))

with col3:
    st.subheader("Output / Activación")
    st.pyplot(plot_output_bar(pred))

# ===================== MATRICES DE PESOS =====================
with st.expander("Ver pesos"):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            st.markdown(f"**Capa {i+1} ({layer.units} neuronas, activación: {layer.activation.__name__})**")
            weights, biases = layer.get_weights()
            st.markdown("**Pesos:** Cada fila corresponde a una neurona de entrada y cada columna a una neurona de salida.")
            st.dataframe(weights)
            st.markdown("**Bias:** Cada valor se suma al output de la neurona correspondiente en la capa.")
            st.dataframe(biases)