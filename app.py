import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Vida Asistanı",
    page_icon="🛠️",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Mobile-like button styling */
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Input/File uploader styling adjustment */
    .stFileUploader {
        padding: 20px;
        border-radius: 15px;
        background-color: #f8f9fa;
        border: 2px dashed #ccc;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🛠️ Vida Asistanı")
st.write("Vida başlığını taratın ve doğru ucu anında öğrenin.")

# --- Helper Functions ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model dosyası ({model_path}) not found! Please place 'best.pt' in the root directory.")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_advice(class_name):
    """
    Returns (Title, Recommendation, HexColor, WarningMessage) based on class name.
    """
    data = {
        'T': ("TORX (YILDIZ)", "T-Serisi Uç (T10-T30)", "#3498db", "Alyan anahtarı zorlamayın."),
        'PH': ("PHILLIPS", "PH Uç (PH1, PH2)", "#9b59b6", "Kesinlikle PZ uç kullanmayın."),
        'PZ': ("POZIDRIV", "PZ Uç (PZ1, PZ2)", "#2ecc71", "Yıldızın arasındaki ince çentiklere dikkat edin."),
        'H': ("HEX (ALYAN)", "Alyan Anahtar (Hex Key)", "#e74c3c", "Köşeleri yuvarlanmış anahtar kullanmayın."),
        'SL': ("DUZ (SLOTTED)", "Düz Tornavida", "#e67e22", "Ucu vida yarığına tam oturtun."),
        'Reference': ("REFERANS NESNE", "Ölçeklendirme", "#95a5a6", "")
    }
    return data.get(class_name, ("Bilinmeyen", "Lütfen tekrar deneyin", "#7f8c8d", ""))

def display_result_card(title, rec, color, warning):
    card_html = f"""
    <div style="
        background-color: {color};
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <h2 style="margin:0; font-size:24px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; margin-bottom: 10px;">{title}</h2>
        <p style="font-size:18px; font-weight:bold; margin-bottom: 5px;">🔧 Önerilen Uç:</p>
        <p style="font-size:20px; margin-top:0;">{rec}</p>
        {f'<div style="background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin-top: 15px;"><strong>⚠️ Uyarı:</strong> {warning}</div>' if warning else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- Main Application Logic ---
model = load_model()

if model:
    tab1, tab2 = st.tabs(["📸 Kamera", "📂 Dosya Yükle"])

    # Image source placeholder
    image_source = None

    with tab1:
        st.header("Kamera ile Tara")
        camera_input = st.camera_input("Fotoğraf Çek")
        if camera_input:
            image_source = camera_input

    with tab2:
        st.header("Resim Yükle")
        uploaded_file = st.file_uploader("Bir resim seçin...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image_source = uploaded_file

    if image_source:
        # Display the input image
        image = Image.open(image_source)
        
        # FIX: Handle mobile image orientation (EXIF data)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass # Keep original if transposition fails

        # st.image(image, caption='İşlenen Görüntü', use_column_width=True)
        
        with st.spinner('Analiz ediliyor...'):
            try:
                # Run inference
                results = model.predict(image, conf=0.45)
                
                # Plot results on image
                # results[0].plot() returns BGR numpy array
                res_plotted = results[0].plot()
                # Convert BGR to RGB for Streamlit
                res_plotted = res_plotted[:, :, ::-1]
                
                # Display the annotated image
                st.image(res_plotted, caption='Tespit Edilen Vidalar', use_container_width=True)
                
                # Check results
                found_detection = False
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        if conf > 0.45:
                            found_detection = True
                            title, rec, color, warning = get_advice(cls_name)
                            display_result_card(title, rec, color, warning)
                
                if not found_detection:
                    st.warning("Görüntüde tanımlı bir vida başı tespit edilemedi veya güven oranı düşük.")
                    
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
