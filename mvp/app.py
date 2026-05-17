import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Handwriting Verification Service",
    page_icon="",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained model with caching."""
    try:
        from model import ModelInference
        
        # Update this path to your actual model file
        MODEL_PATH = 'models/best_cnn_crossvit_c_b_240.pth'
        THRESHOLD = 0.81  # As specified in requirements
        
        model = ModelInference(
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            freeze_backbone='partial'  
        )
        return model
    except Exception as e:
        st.warning(f"Failed to load model: {e}")
        st.info("Using dummy model. Please check the model path.")
        from model import DummyModel
        return DummyModel()


# Title
st.title("Handwriting Verification Service")
st.markdown("### Determine whether two handwriting samples belong to the same author")
st.markdown("---")

# Load model
model = load_model()

# Create two columns for images
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Reference")
    st.caption("Upload the reference handwriting sample")
    reference_file = st.file_uploader(
        "Reference Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        key="reference",
        label_visibility="collapsed"
    )
    
    if reference_file is not None:
        ref_image = Image.open(reference_file)
        st.image(ref_image, caption="Reference Sample", use_column_width=True)
    else:
        st.info("Click 'Browse files' to upload reference image")
        ref_image = None

with col2:
    st.subheader("Questioned")
    st.caption("Upload the questioned handwriting sample")
    questioned_file = st.file_uploader(
        "Questioned Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        key="questioned",
        label_visibility="collapsed"
    )
    
    if questioned_file is not None:
        ques_image = Image.open(questioned_file)
        st.image(ques_image, caption="Questioned Sample", use_column_width=True)
    else:
        st.info("Click 'Browse files' to upload questioned image")
        ques_image = None

# Verify button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    verify_btn = st.button(
        "Verify Handwriting",
        type="primary",
        use_container_width=True,
        disabled=(ref_image is None or ques_image is None)
    )

# Results
if verify_btn and ref_image is not None and ques_image is not None:
    with st.spinner("Analyzing handwriting samples..."):
        result = model.predict(ref_image, ques_image)
    
    st.markdown("---")
    st.subheader("Verification Result")
    
    # Model's decision
    st.markdown("#### Model's decision:")
    
    if result['is_same_author']:
        st.success("**The samples are made by one author**")
    else:
        st.error("**The samples are made by different authors**")
    
    # Probability Assessment (consistent format: decimal)
    st.markdown("---")
    st.markdown("#### Probability Assessment:")
    
    # Display probability and threshold in the same format (decimal, not percentage)
    st.markdown(
        f"The probability that the samples are made by the same author: "
        f"**{result['probability']:.4f}** (or **{result['probability']*100:.2f}%**)"
    )
    
    st.markdown(
        f"The model's probability threshold = **{result['threshold']:.4f}** (or **{result['threshold']*100:.2f}%**, "
        f"{'above' if result['probability'] >= result['threshold'] else 'below'} threshold)"
    )
