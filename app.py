import base64
from flask import Flask, request, render_template, jsonify
import openai
from gtts import gTTS
import os
from dotenv import load_dotenv
# Load environment variables from .env file

load_dotenv()

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("API_KEY") 

@app.route('/')
def index():
    return render_template('index.html')

# New endpoint for transcription only
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    input_language = request.form.get('input_language', 'en')

    # Save uploaded audio file
    audio_file = request.files['audio']
    audio_path = "uploaded_audio.webm"
    audio_file.save(audio_path)

    # Transcribe the audio using OpenAI Whisper
    with open(audio_path, "rb") as audio_data:
        transcript_response = openai.Audio.transcribe("whisper-1", audio_data, language=input_language)
        original_text = transcript_response["text"]

    return jsonify({"original_text": original_text}), 200

# Existing process endpoint for translation and TTS
@app.route('/process', methods=['POST'])
def process_audio():
    original_text = request.form.get('original_text', '')
    output_language = request.form.get('output_language', 'es')
    input_language = request.form.get('input_language', 'hi')  # Get input language

    # Check if original text is a common greeting and handle it
    if original_text.lower() in ["hello", "hi", "hola", "bonjour", "namaste"]:
        translation_prompt = f"Translate this greeting from {input_language} to {output_language}: {original_text}"
    else:
        translation_prompt = f"Translate this from {input_language} to {output_language}: {original_text}"
    
    translation_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": translation_prompt}]
    )
    translated_text = translation_response.choices[0].message['content'].strip()

    # Convert translated text to speech using gTTS
    tts = gTTS(translated_text, lang=output_language)
    output_audio_path = "translated_audio.mp3"
    tts.save(output_audio_path)

    # Encode audio file in base64 for JSON transmission
    with open(output_audio_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    return jsonify({
        "translated_text": translated_text,
        "audio_data": audio_base64
    }), 200


if __name__ == "__main__":
    # Get the PORT environment variable, defaulting to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
