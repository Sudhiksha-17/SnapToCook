import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import numpy as np

# IMPORT BACKEND FUNCTIONS
from recommender import load_data, find_matches, generate_ai_recipe

# Load YOLO models
model1 = YOLO(r"Models\best.pt")
model2 = YOLO(r"Models\yolo_fruits_and_vegetables_v3.pt")

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Chef AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom right, #0e0e0e, #1a1a2e); }
    section[data-testid="stSidebar"] { background-color: #0E1117 !important; border-right: 1px solid #333; }
    h1, h2, h3, h4, h5, h6, p, li, div, span { color: #E0E0E0; }
    label, .stTextInput > label, .stFileUploader > label { color: #E0E0E0 !important; }
    .hero-title { font-size: 5rem !important; font-weight: 900 !important; background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D); -webkit-background-clip: text; -webkit-text-fill-color: transparent !important; margin-bottom: -10px; line-height: 1.1; padding-bottom: 10px; }
    .hero-subtitle { font-size: 1.5rem; font-weight: 400; color: #B0B0B0 !important; margin-top: 0px; }
    .recipe-card { background: rgba(30, 30, 30, 0.6); backdrop-filter: blur(12px); padding: 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 25px; transition: transform 0.3s ease; }
    .recipe-card:hover { transform: translateY(-5px); border: 1px solid #FF914D; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .recipe-title { font-size: 1.6rem; font-weight: 700; color: #FFFFFF !important; margin-bottom: 10px; }
    .missing-item { color: #FF8A8A !important; background-color: rgba(255, 107, 107, 0.15); padding: 4px 10px; border-radius: 6px; margin-right: 6px; display: inline-block; margin-bottom: 6px; font-size: 0.9rem; border: 1px solid rgba(255, 107, 107, 0.2); }
    .match-score { font-size: 1.2rem; font-weight: 800; color: #69F0AE !important; background-color: rgba(0, 230, 118, 0.15); padding: 5px 12px; border-radius: 8px; display: inline-block; margin-bottom: 15px; }
    .stButton>button { background: linear-gradient(90deg, #FF4B4B, #FF914D); color: white !important; border: none; padding: 12px 24px; border-radius: 10px; font-weight: bold; transition: all 0.3s; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
    st.title("👨‍🍳 Chef's Panel")
    st.markdown("---")
    st.info("💡 **Pro Tip:** If you are missing many ingredients, the Generative AI Chef will invent a custom recipe for you.")

# ==========================================
# 3. HELPER FUNCTION
# ==========================================
def detect_ingredients_from_image(image):
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results1 = model1.predict(source=img_array, device="cpu", conf=0.25)
    results2 = model2.predict(source=img_array, device="cpu", conf=0.25)

    def extract_classes(results, model):
        return {model.names[int(cls)] for cls in results[0].boxes.cls}
    
    detected_classes = extract_classes(results1, model1).union(extract_classes(results2, model2))
    detected_classes = [str(c) for c in detected_classes]
    return detected_classes

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
# Hero Section
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown('<h1 class="hero-title">Snap-to-Cook</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">&nbsp;&nbsp;&nbsp;🥦 Your AI-Powered Kitchen Assistant</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []
if 'extra_items' not in st.session_state:
    st.session_state.extra_items = []

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload your fridge photo...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Upload", use_container_width=True)
    
    if not st.session_state.detected_items:
        with st.status("🤖 Scanning image...", expanded=True) as status:
            st.write("Identifying objects...")
            st.session_state.detected_items = detect_ingredients_from_image(image)
            status.update(label="Detection Complete!", state="complete", expanded=False)

# --- ADD EXTRA INGREDIENTS ---
extra_item = st.text_input("➕ Add missing item:", placeholder="e.g. spinach")
if extra_item and extra_item not in st.session_state.detected_items + st.session_state.extra_items:
    st.session_state.extra_items.append(extra_item)

# Combine detected + extra items for recipe search
all_ingredients = st.session_state.detected_items + st.session_state.extra_items

st.success(f"Found {len(all_ingredients)} ingredients!")
st.markdown("##### Ingredients considered:")
st.write(", ".join([f"`{x}`" for x in all_ingredients]))

# --- FIND RECIPES ---
if all_ingredients:
    st.markdown("---")
    st.subheader("🍽️ Recommended for You")
    
    df = load_data()
    matches = find_matches(all_ingredients, df)
    
    if matches and matches[0]['Match Score'] > 0.3:
        for i in range(0, len(matches), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(matches):
                    recipe = matches[i+j]
                    with cols[j]:
                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                        c_img, c_text = st.columns([1, 2])
                        with c_img:
                            img_path = os.path.join("FoodDataset", "Food Images", recipe['Image File'])
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                        with c_text:
                            st.markdown(f"<div class='recipe-title'>{recipe['Recipe']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='match-score'>{int(recipe['Match Score']*100)}% Match</div>", unsafe_allow_html=True)
                        if recipe['Missing']:
                            st.markdown("**⚠️ You need:**")
                            missing_html = ""
                            for item in recipe['Missing'][:5]:
                                clean_item = str(item).replace("['", "").replace("']", "").lower()
                                missing_html += f"<span class='missing-item'>{clean_item}</span>"
                            st.markdown(missing_html, unsafe_allow_html=True)
                        else:
                            st.markdown("✅ **You have everything!**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        if st.button(f"Cook 🍳", key=f"btn_{i+j}"):
                            with st.expander("📝 View Instructions", expanded=True):
                                st.write("Instructions would be loaded here...")
    else:
        st.warning("⚠️ No perfect cookbook matches found.")
        st.info("✨ Activating Generative AI Chef...")
        with st.spinner("👨‍🍳 Chef GPT is inventing a recipe..."):
            ai_recipe_text = generate_ai_recipe(all_ingredients)
        st.markdown("### 🤖 AI Chef's Creation:")
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 25px; border-radius: 15px; border: 1px solid #FF914D; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <div style="color: #E0E0E0; font-size: 1.1rem; line-height: 1.6;">
                {ai_recipe_text.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
