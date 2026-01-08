import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Environment Variables
# =========================
NLU_API_KEY = os.getenv("NLU_API_KEY")
NLU_URL = os.getenv("NLU_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================
# IBM Watson NLU Setup
# =========================
nlu = None
if NLU_API_KEY and NLU_URL:
    try:
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        from ibm_watson import NaturalLanguageUnderstandingV1
        from ibm_watson.natural_language_understanding_v1 import (
            Features,
            SentimentOptions,
            EmotionOptions,
        )

        authenticator = IAMAuthenticator(NLU_API_KEY)
        nlu = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator
        )
        nlu.set_service_url(NLU_URL)
    except Exception:
        nlu = None


# =========================
# Translation Setup
# =========================
USE_GPU = False
gpu_translate = None

try:
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    if torch.cuda.is_available():
        USE_GPU = True
        model_name = "facebook/m2m100_418M"
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to("cuda")

        def gpu_translate(text):
            tokenizer.src_lang = "auto"
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=768
            ).to("cuda")

            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id("en")
            )
            return tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            )
except Exception:
    USE_GPU = False
    gpu_translate = None


# =========================
# Gemini Translation
# =========================
def gemini_translate(text):
    if not GEMINI_API_KEY:
        return text

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"Translate the following text to English:\n{text}"
        )
        return response.text
    except Exception:
        return text


def translate_text(text):
    if USE_GPU and gpu_translate:
        return gpu_translate(text)
    return gemini_translate(text)


# =========================
# Text Analysis
# =========================
def analyze_text(text):
    if not nlu:
        return 0.0, {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "disgust": 0.0
        }

    try:
        response = nlu.analyze(
            text=text,
            features=Features(
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()
            )
        ).get_result()

        sentiment = response["sentiment"]["document"]["score"]
        emotions = response["emotion"]["document"]["emotion"]

        return sentiment, emotions
    except Exception:
        return 0.0, {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "disgust": 0.0
        }
