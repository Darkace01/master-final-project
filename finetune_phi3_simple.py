"""
Simple Phi-3 Fine-tuning Script

A simplified version for quick fine-tuning with sensible defaults.
Suitable for getting started with Phi-3 model adaptation.
"""

import json
import torch, gc
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model


class Phi3Trainer(Trainer):
    """Custom trainer for Phi-3 to handle forward pass compatibility."""

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """Override compute_loss to remove cache-related keys and ensure correct return type."""
        inputs.pop("past_key_values", None)
        inputs.pop("use_cache", None)

        # Call base implementation
        loss = super().compute_loss(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
            return_outputs=return_outputs,
        )

        # If evaluation requested outputs, ensure we return (loss, outputs)
        if return_outputs:
            # Base class returns (loss, outputs) in this mode
            return loss

        # Training/eval with prediction_loss_only -> just return scalar loss
        return loss


def find_all_linear_names(model):
    """Find all linear layer names in the model for LoRA."""
    linear_names = []
    for name, module in model.named_modules():
        if "Linear" in module.__class__.__name__:
            names = name.split(".")
            linear_names.append(names[0] if len(names) == 1 else names[-1])
    return list(set(linear_names))


class Phi3TrainerCallback:
    """Callback to handle Phi-3 specific training issues."""

    def __call__(self, model):
        """Wrap model forward to handle past_key_values."""
        # Disable cache during training to avoid compatibility issues
        model.config.use_cache = False

        # Also patch the model to ignore past_key_values during forward
        original_forward = model.forward

        def forward_with_fix(*args, **kwargs):
            # Remove past_key_values if present to avoid compatibility issues
            kwargs.pop("past_key_values", None)
            kwargs.pop("use_cache", None)
            return original_forward(*args, **kwargs)

        model.forward = forward_with_fix
        return model


def main():
    """Main fine-tuning pipeline."""

    print("=" * 80)
    print("SIMPLE PHI-3 FINE-TUNING")
    print("=" * 80)

    # Configuration
    MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
    TRAINING_DATA_PATH = "data/phi3_enhanced_training_data.jsonl"
    OUTPUT_DIR = "drive/MyDrive/phi3_finetuned_simple"

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load training data
    print(f"\n1. Loading training data from {TRAINING_DATA_PATH}...")
    data = []
    with open(TRAINING_DATA_PATH, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"   ✓ Loaded {len(data)} examples")

    # Load tokenizer
    print(f"\n2. Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded")

    # Load model with 8-bit quantization
    print(f"\n3. Loading model {MODEL_NAME} with 8-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    print("   ✓ Model loaded")

    # Setup LoRA
    print("\n4. Configuring LoRA for parameter-efficient fine-tuning...")
    # Dynamically find target modules for Phi-3
    target_modules = find_all_linear_names(model)
    print(f"   Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\n5. Preparing dataset...")

    def format_messages(example):
        """Format messages into text."""
        messages = example.get("messages", [])
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<{role}>{content}</{role}>\n"
        return {"text": text}

    # Format and tokenize
    formatted_data = [format_messages(ex) for ex in data]
    dataset = Dataset.from_list(formatted_data)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=852,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Split into train/eval
    split_data = tokenized_dataset.train_test_split(test_size=0.05)
    print(f"   ✓ Training set: {len(split_data['train'])} examples")
    print(f"   ✓ Validation set: {len(split_data['test'])} examples")

    # Training arguments
    print("\n6. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        lr_scheduler_type="linear",
        warmup_steps=100,
        logging_steps=50,
        save_steps=200,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=False,
        fp16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_dir="./logs",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    print("   ✓ Training arguments configured")

    # Create trainer
    print("\n7. Creating trainer...")

    # Apply Phi-3 compatibility fix
    callback = Phi3TrainerCallback()
    model = callback(model)

    trainer = Phi3Trainer(
        model=model,
        args=training_args,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print("   ✓ Trainer ready")

    # Train
    print("\n8. Starting training (this may take a while)...")
    print("-" * 80)
    results = trainer.train()
    print("-" * 80)

    # Save model
    print(f"\n9. Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("   ✓ Model saved")

    # Summary
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(
        f"""
Training Results:
  • Final Training Loss: {results.training_loss:.4f}
  • Training Time: {results.training_time:.2f} seconds
  
Model Location: {OUTPUT_DIR}
  • Use this directory for inference
  • Contains model weights and tokenizer

Next Steps:
  1. Load model for inference: AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')
  2. Test with financial customer profiles
  3. Evaluate recommendation quality
  4. Deploy to production
"""
    )


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
