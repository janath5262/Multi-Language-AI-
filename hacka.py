# hacka_multilingual.py
import os
import io
import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3  
from gtts import gTTS
import google.generativeai as genai

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage


os.environ["GOOGLE_API_KEY"] = "ADD_YOUR API_KEY"


genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", None))

# Vision model object (Gemini multimodal)
vision_model = genai.GenerativeModel("models/gemini-2.5-flash")


# OCR + IMAGE PREPROCESS

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change to your path if needed

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Convert to grayscale and apply simple thresholding to improve OCR."""
    try:
        import numpy as np
        import cv2
    except Exception:
        
        return image
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # Use OTSU thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def ocr_tool(image: Image.Image, do_preprocess: bool = True) -> str:
    try:
        img = preprocess_for_ocr(image) if do_preprocess else image
        text = pytesseract.image_to_string(img)
        return text.strip() or "No readable text found."
    except Exception as e:
        return f"OCR error: {e}"


# TTS: create mp3 bytes via gTTS and return bytes

def text_to_mp3_bytes(text: str, lang_code: str = "en") -> bytes:
    """
    Create an mp3 audio of the provided text in the requested language using gTTS.
    Returns bytes suitable for streamlit.audio.
    """
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        
        return b""


# Gemini Vision helper

def gemini_describe_image(image: Image.Image, prompt: str) -> str:
    """
    Send the image bytes first, then the textual prompt to Gemini Vision via the SDK.
    Returns the textual response (response.text).
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    response = vision_model.generate_content(
        [
            {"mime_type": "image/png", "data": img_bytes},
            prompt
        ],
        
    )
    
    return getattr(response, "text", str(response))


# LangChain Safety & Navigation tools

def assess_risk_tool(scene_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0)
    prompt = (
        "You are SafetyAgent. Based ONLY on the scene description below, produce:\n"
        "1) A short bullet list of hazards (max 6)\n"
        "2) Risk level for each hazard (low/medium/high)\n"
        "3) Immediate simple actions to reduce risk\n\n"
        f"SCENE:\n{scene_text}\n\nReturn plain, concise text."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))

def navigation_tool(context_text: str) -> str:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0)
    prompt = (
        "You are NavigationAgent. Using the scene and the identified risks below, produce:\n"
        "1) A concise 2-6 step safe navigation plan (simple actions)\n"
        "2) Up to two alternative safety suggestions\n\n"
        f"INPUT:\n{context_text}\n\nKeep it simple and actionable."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


# LangChain agent initialization

safety_tools = [
    Tool(
        name="AssessRisk",
        func=lambda txt: assess_risk_tool(txt),
        description="Analyze hazards from scene text and return mitigations."
    )
]

navigation_tools = [
    Tool(
        name="CreateNavigation",
        func=lambda ctx: navigation_tool(ctx),
        description="Create navigation steps from scene + safety"
    )
]


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

safety_agent = initialize_agent(
    tools=safety_tools,
    llm=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0),
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=False,
)

navigation_agent = initialize_agent(
    tools=navigation_tools,
    llm=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0),
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=False,
)


# Translation helper (LLM-based)

def translate_text_via_llm(text: str, target_language_name: str) -> str:
    """
    Use the LLM to translate the `text` into the requested language by name.
    This avoids relying on external translate packages and keeps everything in Gemini.
    target_language_name examples: "English", "Hindi", "Spanish", "French", "Tamil"
    """
    if not text or text.strip() == "":
        return ""
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0)
    prompt = (
        f"Translate the following text into {target_language_name}. Keep the meaning exact, "
        "preserve lists/bullets (if any) and do not add or remove content. "
        "Return only the translated text.\n\n"
        f"TEXT:\n{text}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))

# Map UI language code for gTTS (gTTS uses ISO short codes)
LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Tamil": "ta",
    "German": "de",
    "Portuguese": "pt",
    "Chinese (Simplified)": "zh-cn",
    "Arabic": "ar",
    "Bengali": "bn",
    # add more if needed and supported by gTTS
}


# Controller orchestration

def controller_process(image: Image.Image):
    """
    1) Vision: Gemini Vision -> scene_text
    2) OCR: pytesseract -> ocr_text
    3) Safety Agent: safety_agent.run(scene_text)
    4) Navigation Agent: navigation_agent.run(scene + safety)
    Returns dict of outputs.
    """
    vision_prompt = (
        "You are an AI assistant describing images for visually impaired users.\n\n"
        "Describe ONLY what is visible in the image.\n"
        "- List main objects and approximate positions (left/right/center/top/bottom/foreground/background)\n"
        "- Describe environment (indoor/outdoor, lighting if evident)\n"
        "- Note any possible hazards (obstacles, edges, wet floor, traffic, stairs, animals, etc.)\n"
        "- Suggest what a person should do next for safety (concise)\n\n"
        "Be factual and image-grounded. Avoid speculative statements."
    )

    scene_text = gemini_describe_image(image, vision_prompt)
    ocr_text = ocr_tool(image, do_preprocess=True)

    safety_output = safety_agent.run(f"Scene:\n{scene_text}\n\nIdentify hazards and mitigations.")
    nav_input = f"Scene:\n{scene_text}\n\nSafety:\n{safety_output}\n\nProvide concise navigation steps."
    navigation_output = navigation_agent.run(nav_input)

    return {
        "scene": scene_text,
        "ocr": ocr_text,
        "safety": safety_output,
        "navigation": navigation_output
    }


# Streamlit UI

st.set_page_config(page_title="Multi-Language Accessibility Agent", layout="wide")
st.title("🌍 Multi-Language AI Accessibility Agent")

st.markdown(
    "Upload an image. The system will describe the scene, extract text, analyze hazards, "
    "produce navigation instructions, translate into your language of choice, and speak it aloud."
)

col1, col2 = st.columns([1, 3])
with col1:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    lang_choice = st.selectbox("Choose language for translation & TTS",
                               list(LANG_CODE_MAP.keys()), index=0)
    run_btn = st.button("Run Pipeline")
    speak_btn = st.button("🔊 Speak Translated Full Summary")
    do_preprocess = st.checkbox("Preprocess image for better OCR", value=True)
with col2:
    st.info("Outputs will appear here after you run the pipeline.")
    output_area = st.empty()

if uploaded_file is None:
    st.warning("Please upload an image to analyze.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if run_btn:
        with st.spinner("Running Vision → OCR → Safety → Navigation ..."):
            outputs = controller_process(image)

        # Display raw outputs
        output_area.subheader("🔍 Scene Description (Vision)")
        output_area.write(outputs["scene"])

        st.subheader("📄 OCR Text")
        # persist in a text_area with key so it doesn't clear
        st.text_area("Extracted Text:", value=outputs["ocr"], height=180, key="ocr_result")

        st.subheader("⚠️ Safety Assessment (Agent)")
        st.write(outputs["safety"])

        st.subheader("🧭 Navigation Instructions (Agent)")
        st.write(outputs["navigation"])

        # Translate each piece into chosen language
        st.markdown("---")
        st.subheader(f"🌐 Translated outputs — {lang_choice}")

        
        if lang_choice != "English":
            with st.spinner(f"Translating outputs to {lang_choice}..."):
                scene_translated = translate_text_via_llm(outputs["scene"], lang_choice)
                ocr_translated = translate_text_via_llm(outputs["ocr"], lang_choice)
                safety_translated = translate_text_via_llm(outputs["safety"], lang_choice)
                navigation_translated = translate_text_via_llm(outputs["navigation"], lang_choice)
        else:
            scene_translated = outputs["scene"]
            ocr_translated = outputs["ocr"]
            safety_translated = outputs["safety"]
            navigation_translated = outputs["navigation"]

        st.markdown("**Scene (translated):**")
        st.write(scene_translated)

        st.markdown("**OCR (translated):**")
        st.text_area("OCR Translated:", value=ocr_translated, height=140, key="ocr_translated")

        st.markdown("**Safety (translated):**")
        st.write(safety_translated)

        st.markdown("**Navigation (translated):**")
        st.write(navigation_translated)

        # prepare full voice output in target language
        full_translated = (
            "Scene Description: " + scene_translated + "\n\n" +
            "Safety Notes: " + safety_translated + "\n\n" +
            "Navigation: " + navigation_translated + "\n\n" +
            "OCR Text: " + ocr_translated
        )

        # Store audio bytes in session_state so Speak button can play them later
        lang_code = LANG_CODE_MAP.get(lang_choice, "en")
        mp3_bytes = text_to_mp3_bytes(full_translated, lang_code=lang_code)
        if mp3_bytes:
            st.session_state["last_audio"] = mp3_bytes
            st.success("Translation complete — audio ready (press Speak to play).")
        else:
            st.error("Could not generate audio (gTTS may have failed).")

    # Play audio when speak_btn is clicked 
    if speak_btn:
        mp3 = st.session_state.get("last_audio", None)
        if mp3:
            st.audio(mp3, format="audio/mp3")
        else:
            st.warning("No audio ready. Run the pipeline first (Run Pipeline).")
