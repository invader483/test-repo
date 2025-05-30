import sys

# It's good practice to add the current directory to sys.path if needed,
# though for simple structures, it might not be strictly necessary.
# This helps ensure that Python can find the modules.
sys.path.append('.')

# Attempt to import functions and models from existing scripts
try:
    from transcribe_audio import transcribe, model as whisper_model, processor as whisper_processor
    # Check if whisper model loaded (it prints messages and can exit)
    if whisper_model is None or whisper_processor is None: # whisper_model itself might not be None if only processor failed
        print("Error: Whisper model or processor from transcribe_audio.py did not load correctly. Exiting.")
        sys.exit(1) # Exit if critical components are missing
except ImportError as e:
    print(f"Error importing from transcribe_audio.py: {e}. Make sure the file exists and is in the Python path.")
    sys.exit(1)
except SystemExit: # Catch if transcribe_audio.py called exit()
    print("Exiting: Whisper model loading failed in transcribe_audio.py.")
    sys.exit(1)


try:
    from diarize_transcript_llm import diarize_with_llm, model as llm_diarizer_model, tokenizer as llm_diarizer_tokenizer
    if llm_diarizer_model is None or llm_diarizer_tokenizer is None:
        print("Error: LLM diarizer model or tokenizer from diarize_transcript_llm.py did not load correctly. Exiting.")
        sys.exit(1) # Exit if critical components are missing
except ImportError as e:
    print(f"Error importing from diarize_transcript_llm.py: {e}. Make sure the file exists and is in the Python path.")
    sys.exit(1)
except SystemExit: # Catch if diarize_transcript_llm.py called exit() during model load
    print("Exiting: LLM model loading failed in diarize_transcript_llm.py.")
    sys.exit(1)

def run_full_pipeline(audio_path):
    """
    Runs the full audio processing pipeline: transcription then LLM-based diarization.
    """
    print(f"\n--- Starting Full Audio Processing Pipeline for: {audio_path} ---")

    # 1. Transcription
    print("\nStep 1: Transcribing audio...")
    transcript_text = None
    try:
        transcript_text = transcribe(audio_path)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        # transcribe() already has internal error printing, so this is a catch-all

    if transcript_text:
        print("\n--- Raw Transcript ---")
        print(transcript_text)
        print("--- End of Raw Transcript ---")

        # 2. Diarization
        print("\nStep 2: Performing LLM-based diarization...")
        diarized_text = None
        try:
            diarized_text = diarize_with_llm(transcript_text)
        except Exception as e:
            print(f"An error occurred during LLM diarization: {e}")
            # diarize_with_llm() also has internal error printing

        if diarized_text:
            print("\n--- Diarized Transcript (LLM) ---")
            print(diarized_text)
            print("--- End of Diarized Transcript ---")
            return diarized_text
        else:
            print("\nLLM-based diarization failed or returned no output.")
            return None # Return None as diarization failed
    else:
        print("\nTranscription failed or returned no output. Skipping diarization.")
        return None # Return None as transcription failed

if __name__ == "__main__":
    input_audio_file = "input_audio.wav"  # User needs to ensure this file exists

    print(f"Processing audio file: {input_audio_file}")
    print("Ensure the input audio file is a clear voice recording, preferably in WAV format.")
    print("The Whisper model will resample it to 16kHz if needed.")
    
    # Before running, inform about potential Hugging Face Hub token requirements for Gemma
    print("\nNote: The Gemma model ('google/gemma-2b-it') might require authentication "
          "with a Hugging Face Hub token if you haven't used it before or if it's gated.")
    print("If you encounter model download issues for Gemma, run 'huggingface-cli login' "
          "in your terminal and provide your token.\n")

    final_output = None
    try:
        # Check if models were loaded before attempting pipeline
        if whisper_model and llm_diarizer_model:
             final_output = run_full_pipeline(input_audio_file)
        else:
            print("One or more models failed to load. Cannot start the pipeline.")

    except FileNotFoundError:
        # This specific catch might be redundant if transcribe() handles it well,
        # but good for a top-level check.
        print(f"Error: The audio file '{input_audio_file}' was not found. "
              "Please make sure it exists in the current directory or provide the correct path.")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")

    if final_output:
        print("\n--- Full pipeline completed successfully. Final output above. ---")
    else:
        print("\n--- Full pipeline did not complete successfully or produced no output. See messages above. ---")

    print("\nScript finished.")
