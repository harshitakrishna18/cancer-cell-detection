import streamlit as st
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from inference import predict_image

st.set_page_config(page_title="Cancer Cell Detection", page_icon="🔬", layout="centered")

st.title("🔬 PatchCamelyon Cancer Cell Detection")
st.markdown("Upload a histopathology tissue patch (96x96 pixels) to analyze the probability of cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily to run through inference function
    temp_path = "temp_uploaded.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Patch", use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing patch..."):
            prob, tier, advisory = predict_image(temp_path)
            
            if prob is not None:
                prob_percent = prob * 100
                st.metric("Cancer Probability", f"{prob_percent:.2f}%")
                
                # Display advisory with appropriate color block
                if prob_percent > 90:
                    st.error(f"**{tier}**\n\n{advisory}")
                elif prob_percent >= 70:
                    st.warning(f"**{tier}**\n\n{advisory}")
                elif prob_percent >= 40 and prob_percent <= 60:
                    st.info(f"**{tier}**\n\n{advisory}")
                else:
                    st.success(f"**{tier}**\n\n{advisory}")
            else:
                st.error("Error running inference. Please check if the model is trained.")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

st.markdown("---")
st.markdown("> ⚠️ **Disclaimer**: This tool is for research and educational purposes only. It is not a certified medical device and does not replace professional diagnosis.")
