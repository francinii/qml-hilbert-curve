# app.py
import glob
import streamlit as st
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from PIL import Image
import joblib
import os
import sys
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Clasificador de Tumores", page_icon="🧠", layout="centered")

# =============================================================================
# 1. DEFINICIÓN DE LA ARQUITECTURA DEL MODELO (IDÉNTICA AL ENTRENAMIENTO)
# =============================================================================
# Estas constantes deben coincidir con las usadas en el entrenamiento
N_QUBITS= 12
LAYERS = 3
PCA_FEATURES = 12

# Definir el dispositivo cuántico y el QNode
# Usamos 'default.qubit' que es un simulador simple y no requiere dependencias extra pesadas.
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qnode(inputs, weights):
    # Dividimos los qubits para las dos capas cuánticas
    aub = N_QUBITS // 2
    qml.AngleEmbedding(inputs, wires=range(aub))
    qml.BasicEntanglerLayers(weights, wires=range(aub))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(aub)]

weight_shapes = {"weights": (LAYERS, N_QUBITS // 2)}

class HybridModel(nn.Module):
    # La indentación ha sido corregida aquí
    def __init__(self, input_features=PCA_FEATURES):
        super().__init__()
        self.input_features = input_features
        # Capa clásica inicial
        self.clayer_1 = torch.nn.Linear(input_features, input_features)
        # Dos capas cuánticas
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        # Capa clásica final
        self.clayer_2 = torch.nn.Linear(input_features, input_features)
        # Capa final para clasificación binaria
        self.final_layer = torch.nn.Linear(input_features, 1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, self.input_features // 2, dim=1)
        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.clayer_2(x)
        x = self.final_layer(x)
        return x

# =============================================================================
# 2. FUNCIONES DE CARGA Y PREPROCESAMIENTO
# =============================================================================
@st.cache_resource
def load_artifacts(model_path):
    """Carga el modelo y todos los artefactos de preprocesamiento."""
    # Cargar el modelo
    model = HybridModel(input_features=PCA_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Derivar nombres de archivo de los preprocesadores a partir del nombre del modelo
    base_filename = os.path.splitext(os.path.basename(model_path))[0]
    
    # Asumimos que los artefactos están en la carpeta 'preprocessing' y el modelo en 'models'
    # Esta estructura debe coincidir con la de tu script de entrenamiento
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))

    
    scaler_path = next((f for f in pkl_files if 'scaler.pkl' in f and 'angle' not in f), None)
    pca_path = next((f for f in pkl_files if 'pca.pkl' in f), None)
    scaler_angle_path = next((f for f in pkl_files if 'scaler_angle.pkl' in f), None)

    print("scaler_path:", scaler_path)
    print("pca_path:", pca_path)
    print("scaler_angle_path:", scaler_angle_path)

    # Cargar los artefactos (con la corrección del bug de intercambio)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    scaler_angle = joblib.load(scaler_angle_path)
    
    return model, scaler, pca, scaler_angle

def preprocess_image(image_file, scaler, pca, scaler_angle):
    image_size: int = 512,
    """Preprocesa una imagen subida para la predicción."""
    image = Image.open(image_file).convert('L').resize((image_size, image_size))
    img_array = np.array(image)
    flattened_img = img_array.flatten().reshape(1, -1) / 255.0
    scaled_img = scaler.transform(flattened_img)
    pca_img = pca.transform(scaled_img)
    final_features = scaler_angle.transform(pca_img)
    return torch.tensor(final_features, dtype=torch.float32)

# =============================================================================
# 3. INTERFAZ DE LA APLICACIÓN STREAMLIT
# =============================================================================
st.title("🧠 Clasificador de Tumores Cerebrales")
st.write("Sube una imagen de una resonancia magnética para analizarla con un modelo híbrido cuántico-clásico.")

MODELS_DIR = '../models'

try:
    available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    if not available_models:
        st.error(f"No se encontraron modelos (.pt) en el directorio: {MODELS_DIR}")
        st.stop()
except FileNotFoundError:
    st.error(f"El directorio de modelos no existe: '{MODELS_DIR}'. Asegúrate de que la estructura de carpetas sea correcta.")
    st.stop()
selected_model_file = st.selectbox("Selecciona un modelo entrenado:", available_models)
model_path = os.path.join(MODELS_DIR, selected_model_file)


try:
    model, scaler, pca, scaler_angle = load_artifacts(model_path)
    st.success(f"Modelo '{selected_model_file}' y preprocesadores cargados.", icon="✅")
except Exception as e:
    st.error(f"Error al cargar los artefactos para el modelo '{selected_model_file}'.")
    st.error(f"Detalle del error: {e}")
    st.info("Asegúrate de que los archivos .pkl correspondientes existan en la carpeta '../preprocessing'.")
    st.stop()

uploaded_file = st.file_uploader("Elige una imagen para analizar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen Subida", use_column_width=True)
    if st.button("Clasificar Imagen", use_container_width=True, type="primary"):
        with st.spinner("Analizando con el circuito cuántico..."):
            input_tensor = preprocess_image(uploaded_file, scaler, pca, scaler_angle)
            with torch.no_grad():
                output = model(input_tensor)
                probability = torch.sigmoid(output)
                prediction = (probability >= 0.5).int().item()
        
        st.subheader("Resultado del Análisis")
        if prediction == 0: # Asumiendo que 0 es 'tumor'
            st.error("**Resultado: Se detectó un TUMOR.**", icon="❗️")
            confidence = 1 - probability.item()
        else: # Asumiendo que 1 es 'no_tumor'
            st.success("**Resultado: NO se detectó un tumor.**", icon="👍")
            confidence = probability.item()
        
        st.metric(label="Confianza del Modelo", value=f"{confidence:.2%}")
        st.progress(confidence)
