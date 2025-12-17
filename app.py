import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import pytesseract
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# --- Page Configuration ---
st.set_page_config(
    page_title="WeldSentry AI - Professional Weld Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 5px solid #2a5298;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .defect-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .warning-banner {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-banner {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-banner {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('welding_defect_baseline_model.pkl')
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- Image Validation Functions ---
def contains_text(image, threshold=10):
    """Uses OCR to detect if there is significant text in the image."""
    try:
        text = pytesseract.image_to_string(image, timeout=5)
        if len(text.strip()) > threshold:
            return True
        return False
    except Exception:
        return False

def has_sufficient_edge_density(image, threshold=0.015):
    """Checks if the image has enough edge detail to be a weld."""
    try:
        gray_image = np.array(image.convert('L'))
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        edge_pixel_count = np.sum(edges > 0)
        total_pixel_count = edges.size
        edge_density = edge_pixel_count / total_pixel_count
        return edge_density > threshold
    except Exception:
        return True

def is_welding_image(image):
    """Validates if the uploaded image is a welding image."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_np = np.array(image.convert('L'))
        faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return False, "Face Detected - Not a Welding Image"
    except Exception:
        pass

    if contains_text(image):
        return False, "Text Content Detected - Not a Welding Image"

    if not has_sufficient_edge_density(image):
        return False, "Insufficient Detail - Possibly Not a Welding Image"

    return True, "Valid Welding Image"

# --- Feature Extraction Functions ---
def enhance_image(image):
    """Enhances the image using CLAHE for better feature extraction."""
    img_np = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_np)
    return Image.fromarray(enhanced_img).convert('RGB')

def extract_hog_features(image):
    """Extract Histogram of Oriented Gradients features."""
    return hog(image, orientations=9, pixels_per_cell=(8, 8), 
               cells_per_block=(2, 2), transform_sqrt=True)

def extract_lbp_features(image, P=24, R=3):
    """Extract Local Binary Pattern features."""
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm_features(image):
    """Extract Gray-Level Co-occurrence Matrix features."""
    if image.max() == image.min():
        return np.zeros(16)
    
    # Normalize image to 0-255 range
    img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    glcm = graycomatrix(img_normalized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        256, symmetric=True, normed=True)
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])

def get_prediction(image, use_enhancement):
    """Generate prediction for the weld image."""
    if model is None:
        return "Model not loaded", 0.0, None

    img_to_process = image
    if use_enhancement:
        img_to_process = enhance_image(image)

    img_gray = img_to_process.convert('L')
    img_resized = img_gray.resize((128, 128))
    img_array = np.array(img_resized)

    # Extract features
    hog_f = extract_hog_features(img_array)
    lbp_f = extract_lbp_features(img_array)
    glcm_f = extract_glcm_features(img_array)
    features = np.hstack([hog_f, lbp_f, glcm_f])

    # Feature count adjustment (if needed)
    EXPECTED_FEATURES = 8141
    if len(features) != EXPECTED_FEATURES:
        if len(features) > EXPECTED_FEATURES:
            features = features[:EXPECTED_FEATURES]
        else:
            features = np.pad(features, (0, EXPECTED_FEATURES - len(features)))
    
    try:
        pred = model.predict(features.reshape(1, -1))
        proba = model.predict_proba(features.reshape(1, -1))
        confidence = np.max(proba)
        label = "Good Weld" if pred[0] == 0 else "Defective Weld"
        
        return label, confidence, proba[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Prediction error", 0.0, None

# --- Result Display Functions ---
def display_prediction_result(label, confidence, proba=None):
    """Display prediction results with detailed feedback."""
    
    if label == "Good Weld":
        st.markdown(f"""
        <div class="success-banner">
            <h3>✅ Result: {label}</h3>
            <h4>Confidence Score: {confidence:.1%}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("**Analysis:** This weld appears to meet quality standards.")
        
        if confidence >= 0.85:
            st.info("🟢 **High Confidence** - The model is very confident in this classification.")
        elif confidence >= 0.70:
            st.info("🟡 **Moderate Confidence** - The model shows reasonable confidence, but visual inspection is recommended.")
        else:
            st.warning("🟠 **Low Confidence** - The model is uncertain. Professional inspection is strongly recommended.")
        
        st.markdown("""
        **Characteristics of a Good Weld:**
        - Uniform bead width and height
        - Smooth, consistent appearance
        - No visible cracks, pores, or undercuts
        - Proper penetration without excessive reinforcement
        - Clean, free from slag and spatter
        """)
        
    else:  # Defective Weld
        st.markdown(f"""
        <div class="error-banner">
            <h3>❌ Result: {label}</h3>
            <h4>Confidence Score: {confidence:.1%}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.error("**Analysis:** This weld shows potential defects requiring attention.")
        
        if confidence >= 0.85:
            st.info("🔴 **High Confidence** - The model detected clear signs of defects.")
        elif confidence >= 0.70:
            st.info("🟡 **Moderate Confidence** - Potential defects detected. Verification recommended.")
        else:
            st.warning("🟠 **Low Confidence** - The model is uncertain. Professional inspection needed.")
        
        st.markdown("""
        **Common Weld Defects to Check:**
        - **Porosity:** Gas pockets trapped in the weld
        - **Cracks:** Linear fractures in the weld or base metal
        - **Undercut:** Groove at the toe of the weld
        - **Lack of Fusion:** Incomplete bonding between weld and base metal
        - **Slag Inclusion:** Non-metallic material trapped in weld
        - **Overlap:** Excess weld metal extending beyond the toe
        """)
    
    st.markdown("""
    <div class="warning-banner">
        <strong>⚠️ Important Notice:</strong> This is a screening tool developed as an educational project by a mechanical engineering student. 
        Critical welds require verification by certified welding inspectors (CWI). Always follow industry standards (AWS, ASME, ISO) for structural and safety-critical applications.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Understanding Confidence Scores"):
        st.write("""
        **What does the confidence score mean?**
        
        The confidence score indicates how certain the AI model is about its prediction:
        
        - **85-100%:** High confidence - Clear characteristics detected
        - **70-84%:** Moderate confidence - Some ambiguity present
        - **Below 70%:** Low confidence - Significant uncertainty
        
        **Factors affecting confidence:**
        - Image quality and lighting
        - Weld complexity and position
        - Surface conditions and cleanliness
        - Camera angle and distance
        
        **Best practices for accurate results:**
        1. Use well-lit, clear images
        2. Capture the weld from directly above when possible
        3. Ensure the weld surface is clean and visible
        4. Include the entire weld bead in the frame
        """)

# --- Educational Content ---
def show_welding_defects_guide():
    """Display comprehensive welding defects guide."""
    st.markdown("## 📚 Common Welding Defects Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Porosity</h4>
            <p><strong>Description:</strong> Gas pockets or voids in the weld metal</p>
            <p><strong>Causes:</strong> Contaminated base metal, improper shielding gas, moisture</p>
            <p><strong>Prevention:</strong> Clean workpiece, proper gas flow, dry electrodes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Cracks</h4>
            <p><strong>Description:</strong> Linear fractures in weld or heat-affected zone</p>
            <p><strong>Causes:</strong> High cooling rate, residual stress, improper filler</p>
            <p><strong>Prevention:</strong> Preheat material, control cooling, proper technique</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Undercut</h4>
            <p><strong>Description:</strong> Groove melted into base metal at weld toe</p>
            <p><strong>Causes:</strong> Excessive current, wrong angle, fast travel speed</p>
            <p><strong>Prevention:</strong> Reduce current, correct angle, slower travel</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Underfill</h4>
            <p><strong>Description:</strong> Insufficient weld metal in the joint</p>
            <p><strong>Causes:</strong> Low current, fast travel speed, small electrode</p>
            <p><strong>Prevention:</strong> Increase current, slower travel, proper filler</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Overlap</h4>
            <p><strong>Description:</strong> Weld metal extending beyond weld toe without fusion</p>
            <p><strong>Causes:</strong> Low temperature, incorrect angle, excess filler</p>
            <p><strong>Prevention:</strong> Increase heat, correct technique, proper speed</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Slag Inclusion</h4>
            <p><strong>Description:</strong> Non-metallic material trapped in weld</p>
            <p><strong>Causes:</strong> Poor cleaning, incorrect angle, fast travel</p>
            <p><strong>Prevention:</strong> Clean between passes, proper technique</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Lack of Fusion</h4>
            <p><strong>Description:</strong> Incomplete bonding between weld and base metal</p>
            <p><strong>Causes:</strong> Insufficient heat, wrong angle, dirty surface</p>
            <p><strong>Prevention:</strong> Proper current, correct angle, clean surface</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defect-card">
            <h4>🔴 Lack of Penetration</h4>
            <p><strong>Description:</strong> Weld doesn't fully penetrate joint thickness</p>
            <p><strong>Causes:</strong> Low current, fast travel, large root gap</p>
            <p><strong>Prevention:</strong> Increase current, proper joint design, slower speed</p>
        </div>
        """, unsafe_allow_html=True)

def show_welding_quality_indicators():
    """Display quality indicators for good welds."""
    st.markdown("## ✅ Quality Weld Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Visual Indicators</h4>
            <ul>
                <li><strong>Uniform Width:</strong> Consistent bead width throughout</li>
                <li><strong>Smooth Surface:</strong> Free from excessive ripples</li>
                <li><strong>Good Tie-In:</strong> Smooth transition to base metal</li>
                <li><strong>No Spatter:</strong> Minimal or no weld spatter</li>
                <li><strong>Consistent Color:</strong> Even heat tint pattern</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Dimensional Requirements</h4>
            <ul>
                <li><strong>Proper Size:</strong> Adequate but not excessive</li>
                <li><strong>Correct Shape:</strong> Slightly convex profile</li>
                <li><strong>Good Penetration:</strong> Complete joint fusion</li>
                <li><strong>No Undercut:</strong> Flush at weld toes</li>
                <li><strong>Proper Reinforcement:</strong> 1-3mm above surface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Main Application ---
st.markdown("""
<div class="main-header">
    <h1>🔧 WeldSentry AI</h1>
    <h3>Professional Weld Quality Analysis System</h3>
    <p>AI-Powered Screening Tool for Welding Quality Assessment</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("https://via.placeholder.com/250x100/1e3c72/ffffff?text=WeldSentry+AI", use_container_width=True)
    st.markdown("---")
    
    st.header("⚙️ Analysis Settings")
    app_mode = st.radio(
        "Select Analysis Mode:",
        ["Single Image Analysis", "Batch Image Analysis"],
        help="Choose single image for detailed analysis or batch for multiple images"
    )
    
    use_enhancement = st.checkbox(
        "Apply Image Enhancement",
        value=True,
        help="CLAHE enhancement improves feature detection in low-contrast images"
    )
    
    st.markdown("---")
    st.header("📖 Quick Guide")
    st.info("""
    **How to use:**
    1. Upload clear weld image(s)
    2. Review validation results
    3. Check AI prediction
    4. Follow recommendations
    
    **Best results with:**
    - Good lighting
    - Clear focus
    - Clean weld surface
    - Proper camera angle
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p><strong>Version 1.0</strong></p>
        <p>Developed by Mechanical Engineering Student</p>
        <p style='font-size: 0.8em;'>Educational Project - Industrial Training</p>
    </div>
    """, unsafe_allow_html=True)

# Main Content Area
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📚 Defects Guide", "✅ Quality Standards"])

with tab1:
    if app_mode == "Single Image Analysis":
        st.header("Single Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Weld Image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="single_uploader",
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )
        
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            is_valid, reason = is_welding_image(original_image)
            
            st.markdown("### 📸 Image Validation")
            if not is_valid:
                st.error(f"❌ **Validation Failed:** {reason}")
                st.markdown("**Please upload a clear image of a weld bead.**")
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.image(original_image, caption="Invalid Image", width=400)
            else:
                st.success(f"✅ **Validation Passed:** {reason}")
                
                st.markdown("---")
                st.markdown("### 🖼️ Image Preview")
                
                if use_enhancement:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image, caption="Original Image", use_container_width=True)
                    with col2:
                        enhanced_image = enhance_image(original_image)
                        st.image(enhanced_image, caption="Enhanced Image (CLAHE)", use_container_width=True)
                else:
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        st.image(original_image, caption="Original Image", use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🤖 AI Analysis Results")
                
                with st.spinner("🔄 Analyzing weld quality... Please wait..."):
                    label, confidence, proba = get_prediction(original_image, use_enhancement)
                
                display_prediction_result(label, confidence, proba)

    elif app_mode == "Batch Image Analysis":
        st.header("Batch Image Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload Multiple Weld Images",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            key="batch_uploader",
            help="Upload up to 20 images for batch processing"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 20:
                st.warning("⚠️ Maximum of 20 images allowed per batch. Processing first 20 images.")
                uploaded_files = uploaded_files[:20]
            
            st.info(f"📊 Processing {len(uploaded_files)} image(s)...")
            
            # Summary statistics
            good_count = 0
            defect_count = 0
            invalid_count = 0
            
            st.markdown("---")
            st.header("📋 Detailed Analysis Results")
            
            for idx, uploaded_file in enumerate(uploaded_files, 1):
                st.markdown(f"### 🔬 Analysis {idx}/{len(uploaded_files)}: `{uploaded_file.name}`")
                
                original_image = Image.open(uploaded_file)
                is_valid, reason = is_welding_image(original_image)
                
                if not is_valid:
                    st.warning(f"⚠️ Skipping - {reason}")
                    invalid_count += 1
                    st.markdown("---")
                    continue
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="Original", width=350)
                
                if use_enhancement:
                    with col2:
                        enhanced_image = enhance_image(original_image)
                        st.image(enhanced_image, caption="Enhanced", width=350)
                
                with st.spinner(f"Analyzing image {idx}..."):
                    label, confidence, proba = get_prediction(original_image, use_enhancement)
                
                if label == "Good Weld":
                    good_count += 1
                elif label == "Defective Weld":
                    defect_count += 1
                
                display_prediction_result(label, confidence, proba)
                st.markdown("---")
            
            # Display Summary
            st.markdown("## 📊 Batch Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Processed", len(uploaded_files) - invalid_count)
            with col2:
                st.metric("Good Welds", good_count, delta=None, delta_color="normal")
            with col3:
                st.metric("Defective Welds", defect_count, delta=None, delta_color="inverse")
            with col4:
                st.metric("Invalid Images", invalid_count)
            
            if (good_count + defect_count) > 0:
                pass_rate = (good_count / (good_count + defect_count)) * 100
                st.progress(pass_rate / 100)
                st.markdown(f"**Pass Rate:** {pass_rate:.1f}%")

with tab2:
    show_welding_defects_guide()

with tab3:
    show_welding_quality_indicators()
    
    st.markdown("---")
    st.markdown("## 📏 Industry Standards Reference")
    st.markdown("""
    <div class="info-box">
        <h4>Common Welding Standards:</h4>
        <ul>
            <li><strong>AWS D1.1:</strong> Structural Welding Code - Steel</li>
            <li><strong>ASME Section IX:</strong> Welding and Brazing Qualifications</li>
            <li><strong>ISO 5817:</strong> Quality Levels for Imperfections</li>
            <li><strong>AWS D1.2:</strong> Structural Welding Code - Aluminum</li>
            <li><strong>API 1104:</strong> Pipeline Welding Standard</li>
        </ul>
        <p><em>Always consult relevant standards for your specific application.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; background-color: #f8f9fa; padding: 2rem; border-radius: 10px;'>
    <h3>About WeldSentry AI</h3>
    <p><strong>⚠️ Important Disclaimer:</strong></p>
    <p>This is a Version 1.0 prototype model developed for educational purposes during industrial training 
    by a mechanical engineering student. This system serves as a screening tool to assist with initial 
    quality assessment but should NOT be used as the sole method for quality control decisions.</p>
    
    <p><strong>For critical applications:</strong></p>
    <ul style='list-style-position: inside; text-align: center;'>
        <li>Always consult certified welding inspectors (CWI)</li>
        <li>Follow relevant industry standards (AWS, ASME, ISO)</li>
        <li>Perform required destructive and non-destructive testing</li>
        <li>Maintain proper documentation and traceability</li>
    </ul>
    
    <p style='margin-top: 1rem;'><em>Continuous improvement and feedback are essential for model enhancement.</em></p>
</div>
""", unsafe_allow_html=True)