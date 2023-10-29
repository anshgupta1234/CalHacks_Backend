import whisper_timestamped as whisper
from io import BytesIO
import subprocess
import numpy as np
from difflib import SequenceMatcher

def load_audio_from_video(video_file, sr=16000):
    """
    Extract audio from a video file and read it as a mono waveform, resampling as necessary.

    Parameters
    ----------
    video_file: file object
        The video file object received from the API.

    sr: int
        The sample rate to resample the audio if necessary.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # Create a BytesIO object to read the video file
    video_data = video_file.read()
    video_data_io = BytesIO(video_data)

    print(video_data_io.getvalue())

    # This launches a subprocess to extract audio while down-mixing
    # and resampling as necessary. Requires the ffmpeg CLI in PATH.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i", "pipe:0",  # Read input from pipe
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    out = subprocess.run(cmd, input=video_data_io.read(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio_data = np.frombuffer(out.stdout, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data

def get_speech_info(video_path):
    audio = whisper.load_audio(video_path)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return result

def get_sentence_time_segment(sentence, speech_info):
    for segment in speech_info["segments"]:
        if SequenceMatcher(None, sentence, segment["text"]).ratio() > 0.45:
            return (segment["start"], segment["end"])

    return (0, 0)