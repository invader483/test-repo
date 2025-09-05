import torch
import io
#I am god of pythons , i can do anything
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,Helper
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    kbit_trainer
)

# 1. Define Model ID
MODEL_ID = "google/gemma-2b-it" # Switched to 2b-it as 3-1b-it may not exist or be accessible

# Optional: For accessing gated models or private repos, login via CLI:
# huggingface-cli login
# Or in a notebook:
# from huggingface_hub import notebook_login
# notebook_login()

def main():
    # 2. Basic Model Loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # 3. PEFT (LoRA) Setup
    model = prepare_model_for_kbit_training(model)

    # TODO: Verify target_modules for the chosen Gemma model from documentation or examples
    # Common modules for Gemma-like models:
    # 'q_proj', 'k_proj', 'v_proj', 'o_proj', # Attention layers
    # 'gate_proj', 'up_proj', 'down_proj' # MLP layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 4. Placeholder sections
    # 4. Dataset Loading and Preprocessing
    # Expected input: diarized_conversations.jsonl, a JSONL file where each line is a JSON object
    # with a "text" field containing the full diarized conversation string.
    # Example: {"text": "Speaker 1: Hello.\nSpeaker 2: Hi there."}
    dataset = load_dataset("json", data_files="diarized_conversations.jsonl", split="train")

    def preprocess_function(examples):
        # Tokenize the 'text' field (which contains the diarized conversation).
        # For Causal LM, labels are typically the same as input_ids.
        # The Trainer will handle shifting labels if necessary.
        tokenized_inputs = tokenizer(
            examples['text'], # Changed from 'quote' to 'text'
            truncation=True,
            padding='max_length',
            max_length=512 # Ensure this is appropriate for your conversation lengths
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"][:]
        return tokenized_inputs

    # Note: For JSONL, dataset.column_names will likely be just ['text'].
    # remove_columns will then correctly remove the original 'text' column.
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names # Should correctly remove 'text'
    )
    print(f"Dataset loaded and preprocessed. Tokenized dataset size: {len(tokenized_dataset)}")

    # 5. Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./gemma_finetuned_diarized_output", # Changed output directory
        per_device_train_batch_size=1, # Keep low for memory
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1, # For a quick test run
        logging_steps=1,
        save_steps=10, # Adjust based on dataset size, or use save_strategy="epoch"
        report_to="none", # Avoid needing wandb/tensorboard for this example
    )
    print("TrainingArguments defined.")

    # 6. Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # data_collator can be added if specific padding/batching strategies are needed
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 7. Save final model and tokenizer
    print("Saving final model and tokenizer...")
    model.save_pretrained("./gemma_finetuned_diarized_output/final_model") # Changed save path
    tokenizer.save_pretrained("./gemma_finetuned_diarized_output/final_model") # Changed save path
    print("Model and tokenizer saved to ./gemma_finetuned_diarized_output/final_model")

    print("Gemma finetuning script execution complete.")


if __name__ == "__main__":
    main()
    print("Script execution finished successfully.")
