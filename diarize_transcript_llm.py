import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Define Model ID
LLM_MODEL_ID = "google/gemma-2b-it"

# Set device for model inference
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model and Tokenizer (with Quantization)
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically distributes model across available devices
    )
    print(f"LLM model '{LLM_MODEL_ID}' loaded successfully on {model.device}.")
except Exception as e:
    print(f"Error loading LLM model or tokenizer: {e}")
    model = None # Ensure model is None if loading fails
    tokenizer = None

def diarize_with_llm(transcript_text):
    """
    Performs speaker diarization on a transcript using the pre-loaded LLM.
    """
    if not model or not tokenizer:
        print("LLM Model or tokenizer not loaded. Cannot proceed with diarization.")
        return None

    prompt_template = (
        "You are an expert in analyzing conversations. Below is a transcript of a discussion. "
        "Your task is to reformat this transcript by clearly identifying and labeling each speaker. "
        "Use labels like 'Speaker 1:', 'Speaker 2:', etc. "
        "Preserve the entire original text in the new format, attributing each part to a speaker. "
        "If a part of the text seems like a continuation of the previous speaker's turn, keep it with that speaker. "
        "Be consistent with speaker labels. Do not introduce new speakers unnecessarily.\n\n"
        "Original Transcript:\n{transcript_text}\n\n"
        "Reformatted Transcript with Speaker Labels:"
    )
    prompt = prompt_template.format(transcript_text=transcript_text)

    try:
        # Tokenization
        # Ensure tokenizer.pad_token is set if it's None (common for some models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)

        # Generation
        # Adjust max_new_tokens: len of input + estimated length of speaker tags + some buffer
        # This is a heuristic and might need adjustment based on average transcript length and verbosity of speaker tags.
        # Let's assume roughly 15 chars per speaker turn label and estimate number of turns.
        # A simpler approach is a multiplier of input length plus a fixed buffer.
        estimated_max_new_tokens = int(len(input_ids[0]) * 0.5) + 200 # Estimate for speaker labels and structure
        
        print(f"Input token length: {len(input_ids[0])}")
        print(f"Estimated max_new_tokens for generation: {len(input_ids[0]) + estimated_max_new_tokens}")


        outputs = model.generate(
            input_ids,
            max_new_tokens=estimated_max_new_tokens, # Max tokens for the *newly generated part*
            num_beams=3,
            early_stopping=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
        )

        # Decoding
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract Diarized Text
        # The LLM's output includes the prompt. We need to extract only the newly generated part.
        # A robust way is to find the start of the actual diarized response.
        prompt_end_marker = "Reformatted Transcript with Speaker Labels:"
        marker_position = decoded_output.find(prompt_end_marker)

        if marker_position != -1:
            diarized_text = decoded_output[marker_position + len(prompt_end_marker):]
        else:
            # Fallback if the marker is not found (e.g., model didn't follow instructions perfectly)
            # This might include part of the prompt, so it's less ideal.
            print("Warning: Prompt end marker not found in LLM output. Using simple slicing.")
            diarized_text = decoded_output[len(prompt):] # Uses the length of the input prompt string

        return diarized_text.strip()

    except Exception as e:
        print(f"Error during LLM-based diarization: {e}")
        return None

if __name__ == "__main__":
    if model and tokenizer: # Proceed only if model loaded successfully
        sample_transcript = (
            "Hello, how are you today? I'm doing well, thanks for asking! "
            "How about yourself? I'm good too. Just working on this project. "
            "Oh really? What project is that?"
        )
        print("\nOriginal Sample Transcript:\n", sample_transcript)

        try:
            diarized_version = diarize_with_llm(sample_transcript)
            if diarized_version:
                print("\nDiarized Version (LLM):\n", diarized_version)
            else:
                print("LLM Diarization failed or returned empty.")
        except Exception as e:
            print(f"An unexpected error occurred during the main execution: {e}")
    else:
        print("Script cannot run because LLM model or tokenizer failed to load.")

    print("\nScript finished.")
