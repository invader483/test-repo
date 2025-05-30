import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

# Define Model ID
WHISPER_MODEL_ID = "openai/whisper-base"

# Set device for model inference
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model and Processor
try:
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_ID).to(device)
    print(f"Whisper model '{WHISPER_MODEL_ID}' loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    # Exit if model loading fails, as transcription won't be possible
    exit()

def transcribe(audio_path):
    """
    Transcribes an audio file using the pre-loaded Whisper model.
    """
    try:
        # Load the audio file using librosa, resampled to 16kHz
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
        print(f"Audio file '{audio_path}' loaded. Duration: {len(audio_input)/sample_rate:.2f}s")
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}")
        return None

    try:
        # Process the audio
        input_features = processor(
            audio_input,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        # Generate token IDs
        print("Generating transcription...")
        predicted_ids = model.generate(input_features)

        # Decode to text
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        print("Transcription complete.")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    # Define a placeholder audio file path
    audio_file = "input_audio.wav"  # Ensure this file exists or provide a path

    print(f"Attempting to transcribe: {audio_file}")
    try:
        transcription_output = transcribe(audio_file)
        if transcription_output:
            print("\nTranscription:\n", transcription_output)
        else:
            print("Transcription failed or file not found, see messages above.")
    except FileNotFoundError: # This specific catch is somewhat redundant due to internal handling
        print(f"Audio file '{audio_file}' not found. Please provide a valid audio file named '{audio_file}' in the current directory or update the path in the script.")
    except Exception as e:
        print(f"An unexpected error occurred during the transcription process: {e}")

    print("\nScript finished.")
