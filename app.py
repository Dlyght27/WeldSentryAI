
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import webbrowser

# Load the model
try:
    model = joblib.load('welding_defect_baseline_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def is_welding_image(image):
    """
    Placeholder function to detect if an image is a welding image.
    This should be replaced with a more robust detection mechanism.
    """
    # Simple check based on grayscale intensity and color.
    # This is a heuristic and might not be accurate.
    # gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # if np.mean(gray_image) < 50 or np.mean(gray_image) > 200:
    #     return False
    
    # # Check for dominant colors often found in welding (e.g., bright yellows, oranges, blues)
    # hsv = cv2.cvtColor(np.array(image), cv.COLOR_RGB2HSV)
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # if cv2.countNonZero(mask) > image.size[0] * image.size[1] * 0.1:
    #     return True

    return True

def enhance_image(image):
    """
    Enhances the image using contrast limited adaptive histogram equalization (CLAHE).
    """
    img_np = np.array(image.convert('L')) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img_np)
    return Image.fromarray(enhanced_img).convert('RGB')

def get_prediction(image):
    """
    Get prediction and confidence score from the model.
    """
    if model is None:
        return "Model not loaded", 0.0

    # Preprocess the image to match model's expected input
    image = image.resize((224, 224)) # Assuming the model expects 224x224 images
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0) # Add batch dimension

    try:
        prediction = model.predict(image)
        confidence = np.max(model.predict_proba(image))
        label = "Good Weld" if prediction[0] == 1 else "Defective Weld"
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
            label, confidence = get_prediction(enhanced_image)

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
