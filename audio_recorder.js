// audio_recorder.js

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = function (e) {
            const audioBlob = new Blob([e.data], { type: 'audio/wav' });
            const url = URL.createObjectURL(audioBlob);
            const audio = new Audio(url);
            audio.controls = true;
            document.getElementById('audio-container').appendChild(audio);

            // Upload the audio file to the server
            fetch('/upload', {
                method: 'POST',
                body: audioBlob,
                headers: {
                    'Content-Type': 'audio/wav'
                }
            }).then(response => response.text()).then(data => {
                console.log(data);
            }).catch(error => {
                console.error('Error uploading audio:', error);
            });
        };
    }).catch(err => {
        console.error('Error accessing microphone:', err);
    });
}
