import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dense,
    Dropout, Reshape, Multiply, GlobalMaxPooling2D,
    Concatenate, Lambda, Add, Layer
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# ===== AUTO-DOWNLOAD MODEL FILES FROM GOOGLE DRIVE =====
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_CNN_PATH = os.path.join(BASE_PATH, "efficientnet_stenosis_model.h5")
MODEL_HYBRID_PATH = os.path.join(BASE_PATH, "hybrid_cnn_cbam_stenosis_model_lambda_free.h5")

if not os.path.exists(MODEL_CNN_PATH):
    with st.spinner("Downloading EfficientNet CNN model... (this may take a minute)"):
        gdown.download(id="16GmWqH3Wy74AR_oSdYAqRNaaI49TEspg", output=MODEL_CNN_PATH, quiet=False)

if not os.path.exists(MODEL_HYBRID_PATH):
    with st.spinner("Downloading CNN+CBAM model... (this may take a minute)"):
        gdown.download(id="1ZSgXXKVkGhgpBqnrDMG37_TvDOLvGokm", output=MODEL_HYBRID_PATH, quiet=False)

# ===== CUSTOM LAYERS =====

class ReduceMeanLayer(Layer):
    def __init__(self, axis=3, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


class ReduceMaxLayer(Layer):
    def __init__(self, axis=3, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


# --------- CONFIG ---------
GRAPH_PATH = os.path.join(BASE_PATH, "graphs")
CSV_TRAIN = os.path.join(BASE_PATH, "augmented_data", "train", "train.csv")
CSV_VAL = os.path.join(BASE_PATH, "augmented_data", "val", "val.csv")

MODEL_CNN = MODEL_CNN_PATH
MODEL_HYBRID = MODEL_HYBRID_PATH

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4

# --------- CBAM ARCHITECTURE ---------
def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    def reduce_mean_func(x):
        return tf.reduce_mean(x, axis=3, keepdims=True)
    def reduce_max_func(x):
        return tf.reduce_max(x, axis=3, keepdims=True)
    def output_shape_func(input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
    avg_pool = Lambda(reduce_mean_func, output_shape=output_shape_func)(input_feature)
    max_pool = Lambda(reduce_max_func, output_shape=output_shape_func)(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return Multiply()([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

# --------- DATA LOADING ---------
@st.cache_data
def load_data():
    try:
        df_train = pd.read_csv(CSV_TRAIN)
        df_val = pd.read_csv(CSV_VAL)
        return pd.concat([df_train, df_val], axis=0)
    except Exception as e:
        st.error(f"Error loading CSV files: {str(e)}")
        return pd.DataFrame({
            'stenosis_percentage': [25, 35, 45, 55, 65, 75, 85, 95],
            'severity': ['minimal', 'mild', 'mild', 'moderate', 'moderate', 'severe', 'severe', 'critical'],
            'treatment_strategy': ['lifestyle', 'medication', 'medication', 'intervention', 'intervention', 'surgery', 'surgery', 'emergency'],
            'urgency_level': ['low', 'low', 'medium', 'medium', 'high', 'high', 'urgent', 'emergency'],
            'intervention_type': ['none', 'none', 'medication', 'angioplasty', 'angioplasty', 'bypass', 'bypass', 'emergency_surgery'],
            'medication_intensity': ['none', 'low', 'medium', 'medium', 'high', 'high', 'maximum', 'emergency'],
            'next_follow_up': ['1 year', '6 months', '3 months', '3 months', '1 month', '2 weeks', '1 week', 'immediate'],
            'lifestyle_summary': ['basic diet', 'diet + exercise', 'strict diet', 'supervised exercise', 'restricted activity', 'bed rest', 'hospital care', 'ICU care'],
            'lab_monitoring_summary': ['annual', 'biannual', 'quarterly', 'monthly', 'biweekly', 'weekly', 'daily', 'continuous']
        })

# --------- MODEL LOADING ---------
@st.cache_resource
def load_models():
    cnn_model = None
    hybrid_model = None
    try:
        if os.path.exists(MODEL_CNN):
            cnn_model = load_model(MODEL_CNN, compile=False)
            st.success("EfficientNet CNN model loaded successfully.")
        else:
            st.warning(f"EfficientNet CNN model file not found: {MODEL_CNN}")
    except Exception as e:
        st.error(f"Error loading EfficientNet CNN model: {str(e)}")
    try:
        if os.path.exists(MODEL_HYBRID):
            hybrid_model = load_model(MODEL_HYBRID, compile=False, custom_objects={"ReduceMeanLayer": ReduceMeanLayer, "ReduceMaxLayer": ReduceMaxLayer})
            st.success("CNN+CBAM+Regression model loaded successfully.")
        else:
            st.warning(f"CNN+CBAM model file not found: {MODEL_HYBRID}")
    except Exception as e:
        st.error(f"Error loading CNN+CBAM model: {str(e)}")
    return cnn_model, hybrid_model

# --------- UTILITIES ---------
def get_treatment_recommendation(df_lookup, predicted_severity):
    try:
        subset = df_lookup[df_lookup['severity'].str.lower() == predicted_severity.lower()]
        if not subset.empty:
            return subset.iloc[0][['treatment_strategy', 'urgency_level', 'intervention_type', 'medication_intensity', 'next_follow_up', 'lifestyle_summary', 'lab_monitoring_summary']]
        return None
    except Exception as e:
        st.error(f"Error getting treatment recommendation: {str(e)}")
        return None

def get_severity_from_percentage(percentage):
    if percentage < 30: return 'minimal'
    elif percentage < 50: return 'mild'
    elif percentage < 70: return 'moderate'
    elif percentage < 85: return 'severe'
    else: return 'critical'

def predict_image_cnn(model, uploaded_file):
    if model is None: return 0.0, 'unknown'
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr, verbose=0)[0][0]
        stenosis_percentage = max(0, min(100, float(pred * 100)))
        return stenosis_percentage, get_severity_from_percentage(stenosis_percentage)
    except Exception as e:
        st.error(f"Error during CNN prediction: {str(e)}")
        return 0.0, 'unknown'

def predict_image_hybrid(model, uploaded_file):
    if model is None: return 0.0, 'unknown'
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        predictions = model.predict(arr, verbose=0)
        reg_pred = predictions[0][0][0]
        stenosis_percentage = max(0, min(100, float(reg_pred * 100)))
        return stenosis_percentage, get_severity_from_percentage(stenosis_percentage)
    except Exception as e:
        st.error(f"Error during CNN+CBAM prediction: {str(e)}")
        return 0.0, 'unknown'

# --------- STREAMLIT DASHBOARD ---------
st.set_page_config(page_title="ANGIOPLEX — Stenosis Analysis Dashboard", layout="wide")
st.title("ANGIOPLEX: Angiography Analysis and Stenosis Decision Support Platform")
st.markdown("---")

df_lookup = load_data()
cnn_model, hybrid_model = load_models()

col1, col2 = st.columns(2)
with col1:
    if cnn_model is not None:
        st.success("EfficientNet CNN Model: Ready")
    else:
        st.error("EfficientNet CNN Model: Failed to load")
with col2:
    if hybrid_model is not None:
        st.success("CNN+CBAM+Regression Model: Ready")
    else:
        st.error("CNN+CBAM+Regression Model: Failed to load")

if st.checkbox("Show Model Architecture Summaries"):
    col1, col2 = st.columns(2)
    with col1:
        if cnn_model is not None:
            st.write("**EfficientNet CNN — Model Summary**")
            model_summary = []
            cnn_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
    with col2:
        if hybrid_model is not None:
            st.write("**CNN+CBAM+Regression — Model Summary**")
            model_summary = []
            hybrid_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))

tabs = st.tabs(["Exploratory Data Analysis", "Stenosis Prediction", "Model Information"])

# --------- TAB 1: EDA ---------
with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    st.markdown("Distribution and statistical summary of the training dataset.")

    if not df_lookup.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Stenosis Percentage Distribution**")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_lookup['stenosis_percentage'], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
            ax.set_xlabel('Stenosis Percentage (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Stenosis Percentages')
            ax.grid(True, alpha=0.3)
            mean_val = df_lookup['stenosis_percentage'].mean()
            ax.axvline(mean_val, color='firebrick', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}%')
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.write("**Severity Level Distribution**")
            severity_counts = df_lookup['severity'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
            bars = ax.bar(severity_counts.index, severity_counts.values, color=colors[:len(severity_counts)], edgecolor='white')
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Severity Levels')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        st.write("**Dataset Statistics**")
        col3, col4, col5 = st.columns(3)
        with col3: st.metric("Total Samples", len(df_lookup))
        with col4: st.metric("Mean Stenosis", f"{df_lookup['stenosis_percentage'].mean():.1f}%")
        with col5: st.metric("Maximum Stenosis", f"{df_lookup['stenosis_percentage'].max():.1f}%")

        col6, col7, col8 = st.columns(3)
        with col6: st.metric("Minimum Stenosis", f"{df_lookup['stenosis_percentage'].min():.1f}%")
        with col7: st.metric("Standard Deviation", f"{df_lookup['stenosis_percentage'].std():.1f}%")
        with col8: st.metric("Median Stenosis", f"{df_lookup['stenosis_percentage'].median():.1f}%")

# --------- TAB 2: PREDICTION ---------
with tabs[1]:
    st.subheader("Stenosis Prediction and Analysis")
    st.markdown("Select a model and upload a coronary angiography image for automated analysis.")

    available_models = []
    if cnn_model is not None: available_models.append("EfficientNet CNN")
    if hybrid_model is not None: available_models.append("CNN + CBAM + Regression")

    if not available_models:
        st.error("No models are available. Please check model files.")
        st.stop()

    selected_model = st.radio("Select Model:", available_models)

    if selected_model == "EfficientNet CNN":
        st.info("EfficientNet CNN — Pre-trained on ImageNet, fine-tuned for stenosis regression. Recommended for reliable predictions.")
    else:
        st.info("CNN + CBAM + Regression — Custom architecture with channel and spatial attention. Produces both stenosis percentage and severity classification.")

    uploaded_file = st.file_uploader("Upload Angiography Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            img_details = Image.open(uploaded_file)
            st.write(f"**Dimensions:** {img_details.size[0]} x {img_details.size[1]} px")
            st.write(f"**Mode:** {img_details.mode}")

        with col2:
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner(f"Running inference with {selected_model}..."):
                    if selected_model == "EfficientNet CNN":
                        stenosis_perc, severity = predict_image_cnn(cnn_model, uploaded_file)
                    else:
                        stenosis_perc, severity = predict_image_hybrid(hybrid_model, uploaded_file)

                if severity != 'unknown':
                    st.markdown("---")
                    st.write("**Prediction Results**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Predicted Stenosis", f"{stenosis_perc:.2f}%")
                    with col_b:
                        st.metric("Severity Classification", severity.capitalize())

                    st.write("**Stenosis Severity Scale**")
                    st.progress(min(stenosis_perc / 100, 1.0))
                    st.caption(f"Predicted stenosis: {stenosis_perc:.1f}%  |  Model: {selected_model}")

                    severity_labels = {
                        'minimal': 'Minimal stenosis — routine monitoring recommended.',
                        'mild': 'Mild stenosis — lifestyle modification and medication indicated.',
                        'moderate': 'Moderate stenosis — close monitoring and possible intervention.',
                        'severe': 'Severe stenosis — intervention likely required.',
                        'critical': 'Critical stenosis — urgent clinical review indicated.'
                    }
                    st.info(severity_labels.get(severity, ''))

                    treatment = get_treatment_recommendation(df_lookup, severity)
                    if treatment is not None:
                        st.markdown("---")
                        st.write("**Evidence-Based Treatment Recommendation**")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.write(f"**Treatment Strategy:** {treatment['treatment_strategy']}")
                            st.write(f"**Urgency Level:** {treatment['urgency_level']}")
                            st.write(f"**Intervention Type:** {treatment['intervention_type']}")
                            st.write(f"**Medication Intensity:** {treatment['medication_intensity']}")
                        with col4:
                            st.write(f"**Next Follow-up:** {treatment['next_follow_up']}")
                            st.write(f"**Lifestyle Modification:** {treatment['lifestyle_summary']}")
                            st.write(f"**Laboratory Monitoring:** {treatment['lab_monitoring_summary']}")

                    st.markdown("---")
                    st.warning("MEDICAL DISCLAIMER: This system is intended for research and educational purposes only. It does not constitute medical advice and must not be used as a substitute for professional clinical diagnosis or treatment. All outputs should be reviewed by a qualified healthcare professional.")
                else:
                    st.error("Prediction failed. Please verify the image quality and format.")

# --------- TAB 3: MODEL INFO ---------
with tabs[2]:
    st.subheader("Model Information and Comparison")

    st.write("**Architectural Comparison**")
    comparison_df = pd.DataFrame({
        "Feature": ["Base Architecture", "Attention Mechanism", "Output Type", "Training Status", "Primary Use Case", "Computational Cost"],
        "EfficientNet CNN": ["EfficientNetB0 (ImageNet pre-trained)", "None", "Regression (stenosis %)", "Fully trained", "Fast, reliable inference", "Low"],
        "CNN + CBAM + Regression": ["Custom CNN (4 blocks)", "Channel + Spatial (CBAM)", "Regression + Classification", "Fully trained", "Attention-based analysis", "Moderate"]
    })
    st.table(comparison_df)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**EfficientNet CNN**")
        st.write("""
        - EfficientNetB0 backbone, pre-trained on ImageNet
        - Fine-tuned for coronary stenosis regression
        - Input: 224 x 224 x 3
        - Output: Continuous stenosis percentage (0–100%)
        - Approximately 5.3M parameters
        """)
    with col2:
        st.write("**CNN + CBAM + Regression**")
        st.write("""
        - Custom CNN: four convolutional blocks (64, 128, 256, 512 filters)
        - CBAM applied after each block (channel + spatial attention)
        - Input: 224 x 224 x 3
        - Output: Stenosis percentage + severity class (5 categories)
        - Dual-head architecture with shared feature extractor
        """)

    st.markdown("---")
    st.write("**Training Performance**")

    cnn_acc_path   = os.path.join(GRAPH_PATH, "cnn_accuracy_curve.png")
    cnn_loss_path  = os.path.join(GRAPH_PATH, "cnn_loss_curve.png")
    cnn_cm_path    = os.path.join(GRAPH_PATH, "CNN Confusion Matrix.jpg")
    cnn_roc_path   = os.path.join(GRAPH_PATH, "CNN AUC-ROC Curve.jpg")
    cbam_acc_path  = os.path.join(GRAPH_PATH, "cnn_cbam_accuracy_curve.png")
    cbam_loss_path = os.path.join(GRAPH_PATH, "cnn_cbam_loss_curve.png")
    cbam_mae_path  = os.path.join(GRAPH_PATH, "cnn_cbam_mae_curve.png")
    cbam_cm_path   = os.path.join(GRAPH_PATH, "CNN+CBAM Confusion Matrix.jpg")
    cbam_roc_path  = os.path.join(GRAPH_PATH, "CNN+CBAM AUC-ROC Curve.jpg")

    all_paths = [cnn_acc_path, cnn_loss_path, cnn_cm_path, cnn_roc_path,
                 cbam_acc_path, cbam_loss_path, cbam_mae_path, cbam_cm_path, cbam_roc_path]

    if os.path.exists(cnn_acc_path) or os.path.exists(cnn_loss_path):
        st.write("*EfficientNet CNN — Training Curves*")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cnn_acc_path):
                st.image(Image.open(cnn_acc_path), caption="EfficientNet CNN — Accuracy")
        with col2:
            if os.path.exists(cnn_loss_path):
                st.image(Image.open(cnn_loss_path), caption="EfficientNet CNN — Loss")

    if os.path.exists(cnn_cm_path) or os.path.exists(cnn_roc_path):
        st.write("*EfficientNet CNN — Evaluation Metrics*")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cnn_cm_path):
                st.image(Image.open(cnn_cm_path), caption="EfficientNet CNN — Confusion Matrix", use_container_width=True)
        with col2:
            if os.path.exists(cnn_roc_path):
                st.image(Image.open(cnn_roc_path), caption="EfficientNet CNN — ROC Curve", use_container_width=True)

    if os.path.exists(cbam_acc_path) or os.path.exists(cbam_loss_path):
        st.write("*CNN + CBAM + Regression — Training Curves*")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cbam_acc_path):
                st.image(Image.open(cbam_acc_path), caption="CNN+CBAM — Accuracy")
        with col2:
            if os.path.exists(cbam_loss_path):
                st.image(Image.open(cbam_loss_path), caption="CNN+CBAM — Loss")

    if os.path.exists(cbam_cm_path) or os.path.exists(cbam_roc_path):
        st.write("*CNN + CBAM + Regression — Evaluation Metrics*")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cbam_cm_path):
                st.image(Image.open(cbam_cm_path), caption="CNN+CBAM — Confusion Matrix", use_container_width=True)
        with col2:
            if os.path.exists(cbam_roc_path):
                st.image(Image.open(cbam_roc_path), caption="CNN+CBAM — ROC Curve", use_container_width=True)

    if os.path.exists(cbam_mae_path):
        st.write("*CNN + CBAM + Regression — Regression Performance*")
        st.image(Image.open(cbam_mae_path), caption="CNN+CBAM — Mean Absolute Error")

    if not any(os.path.exists(p) for p in all_paths):
        st.info("Training graphs not found. Ensure graph files are present in the graphs/ directory.")

# --------- SIDEBAR ---------
with st.sidebar:
    st.header("ANGIOPLEX")
    st.markdown("Angiography Analysis and Stenosis Decision Support Platform")
    st.markdown("---")

    st.write("**Model Guide**")
    st.write("""
    **EfficientNet CNN**
    Recommended for fast, reliable stenosis percentage prediction.

    **CNN + CBAM + Regression**
    Recommended for attention-guided analysis with dual output (percentage + severity class).
    """)

    st.markdown("---")
    st.write("**Severity Classification Thresholds**")
    st.write("""
    | Severity  | Stenosis Range |
    |-----------|---------------|
    | Minimal   | < 30%         |
    | Mild      | 30 – 50%      |
    | Moderate  | 50 – 70%      |
    | Severe    | 70 – 85%      |
    | Critical  | > 85%         |
    """)

    if not df_lookup.empty:
        st.markdown("---")
        st.write("**Dataset Summary**")
        st.metric("Total Samples", len(df_lookup))
        st.metric("Mean Stenosis", f"{df_lookup['stenosis_percentage'].mean():.1f}%")

    st.markdown("---")
    st.caption("For research and educational use only. Not for clinical deployment.")
