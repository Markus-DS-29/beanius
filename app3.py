import sys
import os

# Add the directory containing st_audiorec to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'path/to/your/modified/st_audiorec'))

from st_audiorec import st_audiorec

wav_audio_data = st_audiorec()
