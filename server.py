from flask import Flask, request, jsonify
import base64
import io
import soundfile as sf

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_audio():
    data = request.get_json()
    audio_data = data['data'].split(",")[1]
    audio_bytes = base64.b64decode(audio_data)
    
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    
    # Process the audio data (e.g., transcription)
    # Here you can save the audio file or send it to a transcription service
    
    return jsonify(status="success")

if __name__ == "__main__":
    app.run(port=8000)
