import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "./models/t5gemma-2-270m-270m"
# Note: Use the specific checkpoint folder or the final output dir
adapter_path = "./models/t5gemma_sql_results/checkpoint-1250"

print(f"Loading base model and adapters on {device}...")

# 2. Load Base Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Use the same dtype as training
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float16)
model = model.to(torch.float16)

# 3. Load LoRA Adapters
model = PeftModel.from_pretrained(model, adapter_path)
model = model.to(device)
model.eval()


def generate_sql(
    prompt_text,
    table_schema="CREATE TABLE table_name (release_date VARCHAR, title VARCHAR)",
):
    # This matches the Llama-2-SQL-Dataset format
    full_prompt = f"Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request.\n\n### Instruction:\n{prompt_text}\n\n### Input:\n{table_schema}\n\n### Response: "

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. Test it!
if __name__ == "__main__":
    test_query = "What is the release date of the movie Inception?"
    print(f"\nQuery: {test_query}")
    print(f"Generated SQL: {generate_sql(test_query)}")

    print("\n--- Interactive Mode ---")
    while True:
        user_query = input("\nEnter a natural language query (or 'exit' to stop): ")
        if user_query.lower() == "exit":
            break
        print(f"Generated SQL: {generate_sql(user_query)}")
