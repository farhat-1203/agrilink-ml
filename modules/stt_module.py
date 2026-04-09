import io
import re
import asyncio
# import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import speech
from google.oauth2 import service_account

try:
    import sounddevice as sd
except:
    sd = None
# ─────────────────────────────────────────────
# 🔢 TEXT NUMBER → DIGIT CONVERTER
# ─────────────────────────────────────────────

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "ten": 10
}

def convert_text_numbers(text):
    for word, num in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(num), text)
    return text


# ─────────────────────────────────────────────
# 🌾 NLP EXTRACTION
# ─────────────────────────────────────────────

CROPS = [
    "wheat", "rice", "tomato", "onion", "potato",
    "गेहूं", "चावल", "टमाटर", "प्याज",
    "गहू", "भात", "टोमॅटो", "कांदा"
]

UNIT_MAP = {
    "quintal": ["quintal", "quintals", "क्विंटल"],
    "kg": ["kg", "kilogram", "किलो"],
    "ton": ["ton", "tonne", "टन"]
}


def extract_crop_details(text):
    text = text.lower()

    # 🔥 Fix common STT mistakes
    corrections = {
        "making": "quintal",
        "mental": "quintal",
        "kent": "quintal"
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    # Convert words → numbers
    text = convert_text_numbers(text)

    # Crop detection
    crop = next((c for c in CROPS if c in text), None)

    # Quantity
    qty_match = re.search(r'(\d+)', text)
    quantity = int(qty_match.group(1)) if qty_match else None

    # Unit
    unit = "unknown"
    for key, values in UNIT_MAP.items():
        if any(v in text for v in values):
            unit = key
            break

    # Normalize
    if quantity:
        if unit == "quintal":
            quantity_kg = quantity * 100
        elif unit == "ton":
            quantity_kg = quantity * 1000
        else:
            quantity_kg = quantity
    else:
        quantity_kg = None

    return {
        "raw_text": text,
        "crop": crop,
        "quantity": quantity,
        "unit": unit,
        "quantity_kg": quantity_kg
    }


# ─────────────────────────────────────────────
# 🎤 STT CLASS (TTS-STYLE)
# ─────────────────────────────────────────────

class SpeechToText:
    def __init__(self, credentials_path: str):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = speech.SpeechClient(credentials=creds)

        # 🔥 Boost important words
        self.phrases = [
            "quintal", "kg", "kilogram", "ton",
            "tomato", "onion", "potato", "wheat", "rice",
            "क्विंटल", "किलो", "टन",
            "टमाटर", "प्याज", "आलू",
            "गहू", "भात", "कांदा"
        ]

    def record_audio(self, duration=5, sample_rate=16000):
        if sd is None:
            raise Exception("Microphone is not available in deployment")
        print(f"🎤 Recording for {duration} seconds... Speak now!")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()

        print("✅ Recording complete")

        # Save debug file
        write("debug.wav", sample_rate, audio)

        buffer = io.BytesIO()
        write(buffer, sample_rate, audio)

        return buffer.getvalue()

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio = speech.RecognitionAudio(content=audio_bytes)

        speech_context = speech.SpeechContext(
            phrases=self.phrases,
            boost=20.0
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN",
            alternative_language_codes=["hi-IN", "mr-IN"],
            enable_automatic_punctuation=True,
            speech_contexts=[speech_context]
        )

        response = self.client.recognize(config=config, audio=audio)

        text_parts = [
            result.alternatives[0].transcript
            for result in response.results
        ]

        final_text = " ".join(text_parts)
        return final_text.strip()


# ─────────────────────────────────────────────
# 🔧 FACTORY
# ─────────────────────────────────────────────

def get_speech_to_text_module():
    return SpeechToText("elite-name-474914-u3-a3e989e02736.json")


# ─────────────────────────────────────────────
# 🚀 MAIN PIPELINE
# ─────────────────────────────────────────────

async def main():
    stt = get_speech_to_text_module()

    # 🎤 Record
    audio_bytes = stt.record_audio()

    # 📝 Transcribe
    text = await stt.transcribe(audio_bytes)
    print("\n📝 Transcription:", text)

    # 📊 Extract
    details = extract_crop_details(text)
    print("\n📊 Extracted Data:", details)


if __name__ == "__main__":
    asyncio.run(main())