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

# ===== CUSTOM LAYERS FOR TRAINED CBAM MODEL =====

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
            st.success("✅ EfficientNet CNN model loaded successfully!")
        else:
            st.warning(f"EfficientNet CNN model file not found: {MODEL_CNN}")
    except Exception as e:
        st.error(f"Error loading EfficientNet CNN model: {str(e)}")
    try:
        if os.path.exists(MODEL_HYBRID):
            hybrid_model = load_model(MODEL_HYBRID, compile=False, custom_objects={"ReduceMeanLayer": ReduceMeanLayer, "ReduceMaxLayer": ReduceMaxLayer})
            st.success("✅ CNN+CBAM+Regression model loaded successfully!")
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
st.set_page_config(page_title="Stenosis Analysis Dashboard", layout="wide")
st.title("ANGIOPLEX: Angiography Analysis and Stenosis Decision Support Platform")

df_lookup = load_data()
cnn_model, hybrid_model = load_models()

col1, col2 = st.columns(2)
with col1:
    if cnn_model is not None: st.success("🟢 EfficientNet CNN Model: Ready")
    else: st.error("🔴 EfficientNet CNN Model: Failed to load")
with col2:
    if hybrid_model is not None:
        st.success("🟢 CNN+CBAM+Regression Model: Ready")
        st.caption("*Using trained CNN+CBAM+Regression model*")
    else: st.error("🔴 CNN+CBAM+Regression Model: Failed to load")

if st.checkbox("Show Model Architectures"):
    col1, col2 = st.columns(2)
    with col1:
        if cnn_model is not None:
            st.write("### EfficientNet CNN Model Summary")
            model_summary = []
            cnn_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
    with col2:
        if hybrid_model is not None:
            st.write("### CNN+CBAM+Regression Model Summary")
            model_summary = []
            hybrid_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))

tabs = st.tabs(["📊 EDA (Exploratory Data Analysis)", "🔍 Stenosis Prediction", "📈 Model Information"])

# --------- TAB 1: EDA ---------
with tabs[0]:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.markdown("**Understanding the dataset used for training both models**")
    if not df_lookup.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Stenosis Percentage Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_lookup['stenosis_percentage'], bins=20, color='lightblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Stenosis Percentage (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Stenosis Percentages')
            ax.grid(True, alpha=0.3)
            mean_val = df_lookup['stenosis_percentage'].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            ax.legend()
            st.pyplot(fig)
        with col2:
            st.write("### Severity Level Distribution")
            severity_counts = df_lookup['severity'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred']
            bars = ax.bar(severity_counts.index, severity_counts.values, color=colors[:len(severity_counts)])
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Severity Levels')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
            plt.tight_layout()
            st.pyplot(fig)
        st.write("### Dataset Statistics")
        col3, col4, col5 = st.columns(3)
        with col3: st.metric("Total Samples", len(df_lookup))
        with col4: st.metric("Average Stenosis %", f"{df_lookup['stenosis_percentage'].mean():.1f}%")
        with col5: st.metric("Max Stenosis %", f"{df_lookup['stenosis_percentage'].max():.1f}%")
        st.write("### Data Insights")
        col6, col7, col8 = st.columns(3)
        with col6: st.metric("Min Stenosis %", f"{df_lookup['stenosis_percentage'].min():.1f}%")
        with col7: st.metric("Std Deviation", f"{df_lookup['stenosis_percentage'].std():.1f}%")
        with col8: st.metric("Median Stenosis %", f"{df_lookup['stenosis_percentage'].median():.1f}%")

# --------- TAB 2: PREDICTION ---------
with tabs[1]:
    st.subheader("🔍 Stenosis Prediction & Analysis")
    st.markdown("**Choose your preferred model and upload an angiography image for analysis**")
    available_models = []
    if cnn_model is not None: available_models.append("EfficientNet CNN")
    if hybrid_model is not None: available_models.append("CNN + CBAM + Regression")
    if not available_models:
        st.error("❌ No models are available for prediction. Please check your model files.")
        st.stop()
    selected_model = st.radio("**Select Model for Prediction:**", available_models, help="Choose between EfficientNet CNN (trained) or CNN+CBAM+Regression")
    if selected_model == "EfficientNet CNN":
        st.info("🤖 **EfficientNet CNN**: Pre-trained model optimized for medical image analysis - for reliable predictions")
    else:
        st.info("🧠 **CNN+CBAM+Regression**: Recommended Trained attention-based model with CBAM modules")
    uploaded_file = st.file_uploader("Upload Angiography Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            img_details = Image.open(uploaded_file)
            st.write(f"**Image Size:** {img_details.size}")
            st.write(f"**Image Mode:** {img_details.mode}")
        with col2:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing image with {selected_model} model..."):
                    if selected_model == "EfficientNet CNN":
                        stenosis_perc, severity = predict_image_cnn(cnn_model, uploaded_file)
                    else:
                        stenosis_perc, severity = predict_image_hybrid(hybrid_model, uploaded_file)
                if severity != 'unknown':
                    st.success(f"📈 **Predicted Stenosis: {stenosis_perc:.2f}%**")
                    st.info(f"🩺 **Severity Level: {severity.capitalize()}**")
                    st.caption(f"*Prediction made using: {selected_model}*")
                    st.write("**Stenosis Level Visualization:**")
                    st.progress(min(stenosis_perc/100, 1.0))
                    if severity == 'minimal': st.success(f"🟢 **{severity.upper()}** stenosis detected")
                    elif severity == 'mild': st.info(f"🔵 **{severity.upper()}** stenosis detected")
                    elif severity == 'moderate': st.warning(f"🟡 **{severity.upper()}** stenosis detected")
                    elif severity == 'severe': st.error(f"🟠 **{severity.upper()}** stenosis detected")
                    else: st.error(f"🔴 **{severity.upper()}** stenosis detected")
                    treatment = get_treatment_recommendation(df_lookup, severity)
                    if treatment is not None:
                        st.markdown("---")
                        st.subheader("💊 Recommended Treatment Plan")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.markdown(f"**🎯 Treatment Strategy:** {treatment['treatment_strategy']}")
                            st.markdown(f"**⚡ Urgency Level:** {treatment['urgency_level']}")
                            st.markdown(f"**🏥 Intervention Type:** {treatment['intervention_type']}")
                            st.markdown(f"**💊 Medication Intensity:** {treatment['medication_intensity']}")
                        with col4:
                            st.markdown(f"**📅 Next Follow-up:** {treatment['next_follow_up']}")
                            st.markdown(f"**🏃 Lifestyle Changes:** {treatment['lifestyle_summary']}")
                            st.markdown(f"**🔬 Lab Monitoring:** {treatment['lab_monitoring_summary']}")
                    st.markdown("---")
                    st.error("⚠️ **IMPORTANT MEDICAL DISCLAIMER:** This AI analysis is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.")
                else:
                    st.error("❌ Failed to analyze the image. Please ensure the image is clear and try again.")
                    st.info("💡 **Tips:** Make sure the image is a clear angiography scan with good contrast.")

# --------- TAB 3: MODEL INFO ---------
with tabs[2]:
    st.subheader("📈 Model Information & Comparison")
    st.write("### Model Comparison")
    comparison_data = {
        "Feature": ["Architecture Base", "Attention Mechanism", "Output Type", "Training Status", "Best Use Case", "Computational Complexity"],
        "EfficientNet CNN": ["EfficientNetB0 (Pre-trained)", "None", "Single regression output", "Fully trained", "Fast, reliable predictions", "Low"],
        "CNN + CBAM + Regression": ["Custom CNN with CBAM", "Channel + Spatial Attention", "Dual output (regression + classification)", "Fully Trained", "Advanced feature extraction", "High"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🔵 EfficientNet CNN Model")
        st.write("""
        **Architecture Features:**
        - Based on EfficientNetB0 architecture
        - Pre-trained on ImageNet, fine-tuned for stenosis
        - Compound scaling for optimal efficiency
        - Single regression head for stenosis percentage

        **Technical Specs:**
        - Input: 224×224×3 Gray Scale images
        - Output: Single stenosis percentage (0-100%)
        - Parameters: ~5.3M (EfficientNetB0 base)
        """)
    with col2:
        st.write("### 🧠 CNN + CBAM + Regression Model")
        st.write("""
        **Architecture Features:**
        - Custom CNN with 4 convolutional blocks
        - CBAM (Convolutional Block Attention Module)
        - Channel attention for feature importance
        - Spatial attention for location focus
        - Dual output heads (regression + classification)

        **Technical Specs:**
        - Input: 224×224×3 Gray Scale images
        - Output: Stenosis % + Severity classification
        - Attention: Channel + Spatial CBAM blocks
        """)

    # ---- ALL GRAPH PATHS (merged from both versions) ----
    st.write("### Training Performance Comparison")

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

    # EfficientNet CNN training curves
    if os.path.exists(cnn_acc_path) or os.path.exists(cnn_loss_path):
        st.markdown("#### 📈 EfficientNet CNN Training Results")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cnn_acc_path):
                st.image(Image.open(cnn_acc_path), caption="EfficientNet CNN - Training Accuracy")
        with col2:
            if os.path.exists(cnn_loss_path):
                st.image(Image.open(cnn_loss_path), caption="EfficientNet CNN - Training Loss")

    # EfficientNet CNN evaluation
    if os.path.exists(cnn_cm_path) or os.path.exists(cnn_roc_path):
        st.markdown("#### 🎯 EfficientNet CNN Evaluation Results")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cnn_cm_path):
                st.image(Image.open(cnn_cm_path), caption="EfficientNet CNN - Confusion Matrix", use_container_width=True)
        with col2:
            if os.path.exists(cnn_roc_path):
                st.image(Image.open(cnn_roc_path), caption="EfficientNet CNN - AUC-ROC Curve", use_container_width=True)

    # CNN+CBAM training curves
    if os.path.exists(cbam_acc_path) or os.path.exists(cbam_loss_path):
        st.markdown("#### 📈 CNN+CBAM+Regression Training Results")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cbam_acc_path):
                st.image(Image.open(cbam_acc_path), caption="CNN+CBAM - Training Accuracy")
        with col2:
            if os.path.exists(cbam_loss_path):
                st.image(Image.open(cbam_loss_path), caption="CNN+CBAM - Training Loss")

    # CNN+CBAM evaluation
    if os.path.exists(cbam_cm_path) or os.path.exists(cbam_roc_path):
        st.markdown("#### 🎯 CNN+CBAM Evaluation Results")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(cbam_cm_path):
                st.image(Image.open(cbam_cm_path), caption="CNN+CBAM - Confusion Matrix", use_container_width=True)
        with col2:
            if os.path.exists(cbam_roc_path):
                st.image(Image.open(cbam_roc_path), caption="CNN+CBAM - AUC-ROC Curve", use_container_width=True)

    # MAE curve
    if os.path.exists(cbam_mae_path):
        st.markdown("#### 📈 CNN+CBAM Mean Absolute Error")
        st.image(Image.open(cbam_mae_path), caption="CNN+CBAM - Mean Absolute Error Over Training")

    if not any(os.path.exists(p) for p in all_paths):
        st.info("📊 Training history graphs not found. Please ensure your graph files are in the graphs/ folder.")

# --------- SIDEBAR ---------
with st.sidebar:
    st.header("🔍 Quick Model Guide")
    st.write("### Model Selection Tips:")
    st.info("""
    **EfficientNet CNN:**
    ✅ Use for reliable predictions
    ✅ Faster processing
    ✅ Production-ready

    **CNN+CBAM+Regression:**
    ✅ Advanced feature extraction
    ✅ Shows advanced attention features
    """)
    st.write("### Severity Levels:")
    st.success("🟢 **Minimal:** < 30%")
    st.info("🔵 **Mild:** 30-50%")
    st.warning("🟡 **Moderate:** 50-70%")
    st.error("🟠 **Severe:** 70-85%")
    st.error("🔴 **Critical:** > 85%")
    st.write("### Dataset Info:")
    if not df_lookup.empty:
        st.metric("Training Samples", len(df_lookup))
        st.metric("Avg Stenosis", f"{df_lookup['stenosis_percentage'].mean():.1f}%")
    st.markdown("---")
    st.caption("⚠️ For educational use only. Consult medical professionals for diagnosis.")
