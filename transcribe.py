import pydub
import torch
import whisper
import numpy as np
from typing import Iterable

class WhisperASR:
    def __init__(self, model_name: str = "base", prompt: str = None, rate: int = 16000, min_silence: int = 500) -> None:
        self._model = whisper.load_model(model_name)
        self._prompt = prompt

        # Audio parameters
        self._min_silence = min_silence  # in seconds
        self._rate = rate                # in Hz

        # Push Whisper to GPU if one is available
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        print(f"WhisperASR.__init__() - Running Whisper on {self._device.upper()}")

    def _split_on_silence(self, audio: np.ndarray) -> Iterable:
        # Use pydub's AudioSegment to efficiently split on silence
        segment = pydub.AudioSegment(
            data=audio.tobytes(),
            frame_rate=self._rate,
            sample_width=audio.dtype.itemsize,
            channels=1
        )

        intervals = pydub.silence.detect_nonsilent(
            segment,
            min_silence_len=self._min_silence,
            silence_thresh=segment.dBFS - 16
        )

        # Slice audio into chunks based on start and stop times of 'nonsilence' (in milliseconds)
        for start, end in intervals:
            starttime = start / 1000
            endtime = end / 1000

            start_idx = int(self._rate * starttime)
            end_idx = int(self._rate * endtime)

            yield starttime, endtime, audio[start_idx: end_idx]

    def transcribe(self, filepath: str, chunk: bool = True) -> Iterable[str]:
        # Open audio file as numpy ndarray
        audio = whisper.load_audio(filepath)

        # [Optional] Cut up audio into chunks of 'non-silence'
        if chunk:
            chunks = self._split_on_silence(audio)
        else:
            starttime = 0
            endtime = audio.shape[0] / self._rate
            chunks = [(starttime, endtime, audio)]

        # Transcribe chunks one-by-one
        for starttime, endtime, chunk in chunks:

            audio = whisper.pad_or_trim(chunk)
            melfreq = whisper.log_mel_spectrogram(audio).to(self._model.device)

            # Allow speaker to switch codes (e.g. from EN to NL)
            _, probs = self._model.detect_language(melfreq)
            lang = max(probs, key=probs.get)

            options = whisper.DecodingOptions(language=lang, prompt=self._prompt)
            result = whisper.decode(self._model, melfreq, options)
            
            yield starttime, endtime, result.text


if __name__ == '__main__':
    asr = WhisperASR(prompt="air traffic control")
    for starttime, endtime, text in asr.transcribe("full_audio.wav"):
        print(starttime, endtime, text)