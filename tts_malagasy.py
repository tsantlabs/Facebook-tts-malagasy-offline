import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile

MODEL_PATH = r"C:\Users\Tsantech\Documents\DEV\tts-project\facebook-tts-malagasy-offline"

print("Chargement du modèle...")
model = VitsModel.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

text = """
Faly miarahaba antsika rehetra , ity ny tu to antsika androany

"""

inputs = tokenizer(text, return_tensors="pt")

print("Génération de l'audio...")
with torch.no_grad():
    out = model(**inputs).waveform  # tensor possible shapes: (batch, time) ou (batch, channels, time)

waveform = out.detach().cpu().numpy()
print("raw waveform shape:", waveform.shape, "dtype:", waveform.dtype)

# --- Normalisation de la forme ---
# Supprimer la dimension batch si présente
if waveform.ndim == 3:              # (batch, channels, time)
    waveform = np.squeeze(waveform, axis=0)  # -> (channels, time) ou (1, time)
if waveform.ndim == 2:
    # Si c'est (1, N) on veut (N,), si c'est (channels, time) on veut (time, channels)
    if waveform.shape[0] == 1:
        waveform = waveform.squeeze(axis=0)         # -> (N,)
    elif waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
        # probablement (channels, time) -> transpose pour (time, channels)
        waveform = waveform.T

print("processed waveform shape:", waveform.shape)

# Assurer float et borne [-1,1]
waveform = waveform.astype(np.float32)
waveform = np.clip(waveform, -1.0, 1.0)

# Conversion en int16
audio_int16 = (waveform * 32767).astype(np.int16)

# Sampling rate (force int)
sampling_rate = int(getattr(model.config, "sampling_rate", 16000))
print("sampling_rate:", sampling_rate)

output_file = "tts_malagasy.wav"
wavfile.write(output_file, rate=sampling_rate, data=audio_int16)
print(f"✅ Audio sauvegardé dans {output_file}")
