import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# 1. Setup Device (M1 Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Model & Tokenizer
model_id = "./models/t5gemma-2-270m-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load in half-precision (float16) using the new 'dtype' argument
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float16)
model = model.to(torch.float16) # Explicitly force all parts to float16 for MPS compatibility

# 3. Apply LoRA (Massive memory savings for your 16GB Mac)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model = model.to(device) # Move to MPS device
model.print_trainable_parameters()

# 4. Load and Format Dataset
dataset = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset")


def preprocess_function(examples):
    # This dataset version already has the full prompt in the 'input' field
    inputs = [inp.strip() for inp in examples["input"]]
    targets = examples["output"]

    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(text_target=targets, max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize data (using a subset for faster experimentation)
tokenized_dataset = (
    dataset["train"].select(range(5000)).map(preprocess_function, batched=True)
)
tokenized_eval = (
    dataset["eval"].select(range(500)).map(preprocess_function, batched=True)
)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./models/t5gemma_sql_results",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    gradient_checkpointing=True,
    report_to="none",
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# 7. Start the experiment!
print("Starting training loop...")
trainer.train()
