
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

# Load the model
try:
    model = joblib.load('welding_defect_baseline_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def is_welding_image(image):
    """
    Uses a pre-trained face detector to filter out images of people.
    If a face is detected, it's not a welding image.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_np = np.array(image.convert('L'))  # Convert to grayscale numpy array
        faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If any faces are detected, return False
        if len(faces) > 0:
            return False
        return True  # No faces detected
    except Exception as e:
        st.warning(f"Could not perform face detection: {e}")
        # Default to True if face detection fails, to not block processing
        return True

def enhance_image(image):
    """
    Enhances the image using contrast limited adaptive histogram equalization (CLAHE).
    """
    img_np = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_np)
    return Image.fromarray(enhanced_img).convert('RGB')

def extract_hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, visualize=False)
    return features

def extract_lbp_features(image, P=24, R=3):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 2), range=(0, P + 1))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-8)
    return hist

def extract_glcm_features(image):
    if np.max(image) == np.min(image):
        return np.zeros(16)

    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                        symmetric=True, normed=True)
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    glcm_features = np.concatenate([graycoprops(glcm, prop).ravel() for prop in props])
    return glcm_features

def extract_all_features(image):
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    glcm_features = extract_glcm_features(image)
    return np.hstack([hog_features, lbp_features, glcm_features])

def get_prediction(image, use_enhancement):
    if model is None:
        return "Model not loaded", 0.0

    processed_image = image
    if use_enhancement:
        # Convert to grayscale before enhancing
        processed_image = enhance_image(image.convert('L'))

    # Preprocess the image: convert to grayscale, then resize
    img_gray = processed_image.convert('L')
    image_resized = img_gray.resize((128, 128))
    img_array = np.array(image_resized)

    features = extract_all_features(img_array)
    features_reshaped = features.reshape(1, -1)

    try:
        prediction = model.predict(features_reshaped)
        confidence = np.max(model.predict_proba(features_reshaped))
        label = "Good Weld" if prediction[0] == 0 else "Defective Weld"
        return label, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction error", 0.0

# --- UI Display Functions ---
def display_prediction_result(label, confidence):
    st.subheader("Prediction:")
    if label == "Good Weld":
        st.success(f"**Result:** {label} (Confidence: {confidence:.2f})")
    else:
        st.error(f"**Result:** {label} (Confidence: {confidence:.2f})")

# --- Streamlit App ---
st.set_page_config(page_title="WeldSentry AI", layout="wide")
st.title("WeldSentry AI 🤖")
st.markdown("### Advanced Weld Quality Analysis")
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    app_mode = st.radio("Choose Mode", ["Single Image Analysis", "Batch Image Analysis"])
    use_enhancement = st.checkbox("Apply Image Enhancement")

# --- Main App Logic ---
if app_mode == "Single Image Analysis":
    uploaded_file = st.file_uploader("Upload a Weld Image", type=['jpg', 'jpeg', 'webp'], key="single_uploader")

    if uploaded_file:
        original_image = Image.open(uploaded_file)

        if not is_welding_image(original_image):
            st.error("This does not appear to be a welding image. A face was detected. Please upload a relevant image.")
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

            if not is_welding_image(original_image):
                st.warning(f"Skipping image `{uploaded_file.name}` as it does not appear to be a welding image (face detected).")
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
    '''
    **Disclaimer:** This is a prototype model developed by a final year mechanical engineering student. 
    For critical applications, always consult a qualified welding inspector.
    '''
)
