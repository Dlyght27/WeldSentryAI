
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import webbrowser
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
    Placeholder function to detect if an image is a welding image.
    """
    return True

def enhance_image(image):
    """
    Enhances the image using contrast limited adaptive histogram equalization (CLAHE).
    """
    img_np = np.array(image.convert('L')) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img_np)
    return Image.fromarray(enhanced_img).convert('RGB')

def extract_hog_features(image):
    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
    return features

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

def extract_all_features(image):
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    glcm_features = extract_glcm_features(image)
    
    combined_features = np.hstack([hog_features, lbp_features, glcm_features])
    return combined_features

def get_prediction(image):
    """
    Get prediction and confidence score from the model.
    """
    if model is None:
        return "Model not loaded", 0.0

    # Preprocess the image
    image_resized = image.resize((128, 128))
    img_gray = np.array(image_resized.convert('L'))

    # Extract features
    features = extract_all_features(img_gray)

    # Reshape for the model
    features_reshaped = features.reshape(1, -1)

    try:
        prediction = model.predict(features_reshaped)
        confidence = np.max(model.predict_proba(features_reshaped))
        label = "Good Weld" if prediction[0] == 0 else "Defective Weld"
        return label, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction error", 0.0

# --- Streamlit App ---

st.set_page_config(page_title="WeldSentry AI", layout="wide")

st.title("WeldSentry AI 🤖")
st.markdown("### Advanced Weld Quality Analysis")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader(
        "Upload Weld Images", 
        type=["jpg", "jpeg", "webp"], 
        accept_multiple_files=True
    )
    st.info("You can upload up to 20 images at a time.")


if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("Maximum of 20 images allowed per batch.")
        uploaded_files = uploaded_files[:20]

    st.header("Analysis Results")

    for uploaded_file in uploaded_files:
        st.markdown(f"#### Results for `{uploaded_file.name}`")
        col1, col2 = st.columns(2)

        original_image = Image.open(uploaded_file)

        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        
        if not is_welding_image(original_image):
            st.error("This does not appear to be a welding image. Please upload a relevant image.")
            continue

        with col2:
            with st.spinner("Enhancing image..."):
                enhanced_image = enhance_image(original_image)
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

        with st.spinner("Analyzing weld quality..."):
            label, confidence = get_prediction(original_image)

        st.markdown("---")
        st.subheader("Prediction:")
        
        if label == "Good Weld":
            st.success(f"**Result:** {label} (Confidence: {confidence:.2f})")
            st.markdown(
                """
                **Feedback:** The weld appears to be of good quality. 
                - **Appearance:** Consistent bead width and minimal spatter.
                - **Recommendation:** No immediate action required. Continue monitoring for consistency.
                """
            )
        else:
            st.error(f"**Result:** {label} (Confidence: {confidence:.2f})")
            st.markdown(
                """
                **Feedback:** The weld shows potential defects.
                - **Possible Issues:** Could include porosity, cracks, or undercut.
                - **Recommendation:** A detailed inspection by a certified welding inspector is recommended.
                """
            )
        st.markdown("---")


# Footer
st.markdown("---")
st.markdown("### About WeldSentry AI")
st.info(
    """
    **Disclaimer:** This is the first version of the WeldSentry AI model and it is not perfect. 
    It was developed by a final year mechanical engineering student. 
    For critical applications, a qualified welding consultant or inspector should be consulted.
    """
)

st.markdown("---")
col1, col2 = st.columns([1, 4])

with col1:
    st.markdown("Built by:")

with col2:
    st.markdown("Danel Israel")
    st.markdown("danielisrael120@gmail.com")

st.markdown(
    """
    <style>
        .st-emotion-cache-18ni7ap{
            padding-top: 2rem;
        }
        .st-emotion-cache-z5fcl4{
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
