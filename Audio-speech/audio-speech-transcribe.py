import requests
import time
import assemblyai as aai

audio_file_path = r"C:/Users/Amroy/Downloads/converted_audio.wav"
# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai

# Replace with your API key
aai.settings.api_key = "d6e139f50eb94076ad4fb332049809c4"

# URL of the file to transcribe
FILE_URL = "C:/Users/Amroy/Downloads/converted_audio.wav"

# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    print(transcript.text)


