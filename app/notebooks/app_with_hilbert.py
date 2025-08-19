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

# Importar HilbertCurveProcessor y HilbertTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from hilbert import HilbertCurveProcessor, HilbertTransformer

# --- Configuración de la página ---
st.set_page_config(page_title="Clasificador de Tumores con Hilbert", page_icon="🧠", layout="centered")

# =============================================================================
# 1. DEFINICIÓN DE LA ARQUITECTURA DEL MODELO (IDÉNTICA AL ENTRENAMIENTO)
# =============================================================================
# Estas constantes deben coincidir con las usadas en el entrenamiento
# Parámetros del modelo entrenado: 2025-08-02 19:18:44,7790.75,200,0.05,12,3,64,0.6386,0.6673,0.4400,0.5699,59,AUTOENCODER_BC_model_epoch_59_acc_0.6673.pt,128
N_QUBITS = 12  # features
LAYERS = 3
PCA_FEATURES = 12
IMAGE_SIZE = 128

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

# Definir la arquitectura del encoder (debe coincidir con el entrenamiento)
class Encoder(nn.Module):
    def __init__(self, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 256, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)

# =============================================================================
# 2. FUNCIONES DE CARGA Y PREPROCESAMIENTO
# =============================================================================
@st.cache_resource
def load_artifacts(model_path):
    """Carga el modelo, el encoder y el scaler_angle para preprocesamiento."""
    # Cargar el modelo principal
    model = HybridModel(input_features=PCA_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # Cargar el encoder
    base_filename = os.path.splitext(os.path.basename(model_path))[0]
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_with_hilbert'))
    encoder_path = os.path.join(models_dir, f"{base_filename}_encoder.pt")
    encoder = Encoder(latent_dim=12)
    encoder_state = torch.load(encoder_path, map_location=torch.device('cpu'))
    # Adaptar claves si es necesario
    if all(not k.startswith('encoder.') for k in encoder_state.keys()):
        encoder_state = {f'encoder.{k}': v for k, v in encoder_state.items()}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    # Cargar scaler_angle
    pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    scaler_angle_path = next((f for f in pkl_files if 'scaler_angle.pkl' in f), None)
    print("scaler_angle_path:", scaler_angle_path)
    if scaler_angle_path is None:
        raise FileNotFoundError("No se encontró scaler_angle.pkl en la carpeta de modelos.")
    scaler_angle = joblib.load(scaler_angle_path)
    return model, encoder, scaler_angle

def preprocess_image(image_file, encoder, scaler_angle):
    # Procesar la imagen igual que en el entrenamiento (CON Hilbert)
    # 1. Convertir a escala de grises y redimensionar a 128x128
    image = Image.open(image_file).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 2. Aplicar transformación Hilbert directamente usando HilbertTransformer
    hilbert_transformer = HilbertTransformer(2)  # 2D image
    hilbert_img = hilbert_transformer.transform(img_array)
    hilbert_img = hilbert_img.reshape(1, -1)
    
    # 3. Pasar por el encoder
    x = torch.tensor(hilbert_img, dtype=torch.float32).unsqueeze(1)  # (1, 1, 16384)
    with torch.no_grad():
        encoded = encoder(x).numpy()
    
    # 4. Aplicar scaler_angle
    final_features = scaler_angle.transform(encoded)
    return torch.tensor(final_features, dtype=torch.float32)

# =============================================================================
# 3. INTERFAZ DE LA APLICACIÓN STREAMLIT
# =============================================================================
st.title("🧠 Clasificador de Tumores Cerebrales con Curva de Hilbert")
st.write("Sube una imagen de una resonancia magnética para analizarla con un modelo híbrido cuántico-clásico que utiliza la transformación de Hilbert.")

MODELS_DIR = '../models_with_hilbert'

try:
    # Filtrar solo los archivos de modelo principal (excluir _encoder.pt)
    available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt') and not f.endswith('_encoder.pt')]
    if not available_models:
        st.error(f"No se encontraron modelos (.pt) en el directorio: {MODELS_DIR}")
        st.stop()
except FileNotFoundError:
    st.error(f"El directorio de modelos no existe: '{MODELS_DIR}'. Asegúrate de que la estructura de carpetas sea correcta.")
    st.stop()

# Mostrar información del modelo recomendado
st.info(f"**Modelo recomendado:** AUTOENCODER_BC_model_epoch_59_acc_0.6673.pt (Accuracy: 66.73%)")

selected_model_file = st.selectbox("Selecciona un modelo entrenado:", available_models)
model_path = os.path.join(MODELS_DIR, selected_model_file)

try:
    model, encoder, scaler_angle = load_artifacts(model_path)
    st.success(f"Modelo '{selected_model_file}', encoder y scaler_angle cargados.", icon="✅")
    st.info(f"📁 Directorio de modelos: {MODELS_DIR}")
except Exception as e:
    st.error(f"Error al cargar los artefactos para el modelo '{selected_model_file}'.")
    st.error(f"Detalle del error: {e}")
    st.info("Asegúrate de que el archivo scaler_angle.pkl y el encoder correspondiente existan en la carpeta '../models_with_hilbert'.")
    st.info("Los archivos necesarios son:")
    st.info(f"  - {selected_model_file} (modelo principal)")
    st.info(f"  - {selected_model_file.replace('.pt', '_encoder.pt')} (encoder)")
    st.info(f"  - {selected_model_file.replace('.pt', '_scaler_angle.pkl')} (scaler)")
    st.stop()

uploaded_file = st.file_uploader("Elige una imagen para analizar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen Subida", use_column_width=True)
    if st.button("Clasificar Imagen", use_container_width=True, type="primary"):
        with st.spinner("Analizando con el circuito cuántico y transformación de Hilbert..."):
            input_tensor = preprocess_image(uploaded_file, encoder, scaler_angle)
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
        
        # Mostrar información adicional del modelo
        st.subheader("Información del Modelo")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy de Entrenamiento", "66.73%")
        with col2:
            st.metric("Épocas de Entrenamiento", "200")
        with col3:
            st.metric("Learning Rate", "0.05")
