
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import pytesseract
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# --- Model Loading ---
try:
    model = joblib.load('welding_defect_baseline_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- Image Validation Functions ---
def contains_text(image, threshold=10):
    """Uses OCR to detect if there is significant text in the image."""
    try:
        text = pytesseract.image_to_string(image, timeout=5)  # 5-second timeout
        if len(text.strip()) > threshold:
            return True
        return False
    except Exception as e:
        # If Pytesseract is not installed or fails, log a warning and continue
        st.warning(f"Could not perform OCR: {e}. Skipping text check.")
        return False

def has_sufficient_edge_density(image, threshold=0.015):
    """
    Checks if the image has enough edge detail to be a weld.
    Helps filter out simple graphics or blurry/out-of-focus images.
    """
    try:
        gray_image = np.array(image.convert('L'))
        # Use Canny edge detector
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        
        # Calculate the ratio of edge pixels to total pixels
        edge_pixel_count = np.sum(edges > 0)
        total_pixel_count = edges.size
        edge_density = edge_pixel_count / total_pixel_count
        
        return edge_density > threshold
    except Exception as e:
        st.warning(f"Could not perform edge detection: {e}. Skipping check.")
        return True # Default to true to avoid blocking on failure

def is_welding_image(image):
    """Combines multiple checks to validate the uploaded image."""
    # 1. Face Detection (Most reliable check for a common invalid image type)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_np = np.array(image.convert('L'))
        faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return False, "Face Detected"
    except Exception as e:
        st.warning(f"Could not perform face detection: {e}.")

    # 2. Text Detection
    if contains_text(image):
        return False, "Text Content Detected"

    # 3. Edge Density Check (Filters out simple graphics)
    if not has_sufficient_edge_density(image):
        return False, "Low Detail / Simple Graphic Detected"

    return True, "Valid Image"

# --- Feature Extraction & Prediction ---
def enhance_image(image):
    """Enhances the image using CLAHE."""
    img_np = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_np)
    return Image.fromarray(enhanced_img).convert('RGB')

def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)

def extract_lbp_features(image, P=24, R=3):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm_features(image):
    if image.max() == image.min(): return np.zeros(16)
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])

def get_prediction(image, use_enhancement):
    if model is None: return "Model not loaded", 0.0

    img_to_process = image
    if use_enhancement:
        # Pass the original color image to be converted within the function
        img_to_process = enhance_image(image)

    img_gray = img_to_process.convert('L')
    img_resized = img_gray.resize((128, 128))
    img_array = np.array(img_resized)

    # Extract all features
    hog_f = extract_hog_features(img_array)
    lbp_f = extract_lbp_features(img_array)
    glcm_f = extract_glcm_features(img_array)
    features = np.hstack([hog_f, lbp_f, glcm_f])

    try:
        pred = model.predict(features.reshape(1, -1))
        proba = np.max(model.predict_proba(features.reshape(1, -1)))
        label = "Good Weld" if pred[0] == 0 else "Defective Weld"
        return label, proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Prediction error", 0.0

# --- UI Functions ---
def display_prediction_result(label, confidence):
    st.subheader("Prediction:")
    if label == "Good Weld":
        st.success(f"**Result:** {label} (Confidence: {confidence:.2f})")
    else:
        st.error(f"**Result:** {label} (Confidence: {confidence:.2f})")

st.set_page_config(page_title="WeldSentry AI", layout="wide")
st.title("WeldSentry AI 🤖")
st.markdown("### Advanced Weld Quality Analysis")
st.markdown("---")

with st.sidebar:
    st.header("Controls")
    app_mode = st.radio("Choose Mode", ["Single Image Analysis", "Batch Image Analysis"])
    use_enhancement = st.checkbox("Apply Image Enhancement")

# --- Main App Logic ---
if app_mode == "Single Image Analysis":
    uploaded_file = st.file_uploader("Upload a Weld Image", type=['jpg', 'jpeg', 'webp'], key="single_uploader")
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        is_valid, reason = is_welding_image(original_image)

        if not is_valid:
            st.error(f"Invalid Image: {reason}. Please upload a relevant welding image.")
            st.image(original_image, caption="Invalid Image", width=300)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", width=400)
            if use_enhancement:
                with col2:
                    enhanced_image = enhance_image(original_image)
                    st.image(enhanced_image, caption="Enhanced Image", width=400)
            
            with st.spinner("Analyzing weld quality..."):
                label, confidence = get_prediction(original_image, use_enhancement)
            st.markdown("---")
            display_prediction_result(label, confidence)

elif app_mode == "Batch Image Analysis":
    uploaded_files = st.file_uploader("Upload Weld Images", type=['jpg', 'jpeg', 'webp'], accept_multiple_files=True, key="batch_uploader")
    if uploaded_files:
        if len(uploaded_files) > 20:
            st.warning("Maximum of 20 images allowed per batch.")
            uploaded_files = uploaded_files[:20]

        st.header("Analysis Results")
        for uploaded_file in uploaded_files:
            st.markdown(f"#### Results for `{uploaded_file.name}`")
            original_image = Image.open(uploaded_file)
            is_valid, reason = is_welding_image(original_image)

            if not is_valid:
                st.warning(f"Skipping `{uploaded_file.name}`: {reason}.")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", width=300)
            if use_enhancement:
                with col2:
                    enhanced_image = enhance_image(original_image)
                    st.image(enhanced_image, caption="Enhanced Image", width=300)

            with st.spinner(f"Analyzing `{uploaded_file.name}`..."):
                label, confidence = get_prediction(original_image, use_enhancement)
            display_prediction_result(label, confidence)
            st.markdown("---")

# --- Footer ---
st.markdown("### About WeldSentry AI")
st.info(
    '''**Disclaimer:** This is a prototype model developed for educational purposes. 
    For critical applications, always consult a qualified welding inspector.'''
)
