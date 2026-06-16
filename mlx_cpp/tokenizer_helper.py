import sys
import json
import os
from transformers import AutoTokenizer

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
else:
    MODEL_PATH = "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4"

def main():
    try:
        # Load tokenizer with the same settings as python generate_sft_data.py
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
    except Exception as e:
        sys.stderr.write(f"Failed to load tokenizer: {e}\n")
        sys.exit(1)

    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"

    # Main loop reading JSON commands from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            command = req.get("command")

            if command == "encode_chat_prompts":
                problems = req.get("problems", [])
                tokenized = []
                for p in problems:
                    chat = [{"role": "user", "content": prompt_format.format(p)}]
                    # apply chat template
                    tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
                    if hasattr(tokens, "input_ids"):
                        tokens = tokens.input_ids
                    elif isinstance(tokens, dict) and "input_ids" in tokens:
                        tokens = tokens["input_ids"]
                    tokenized.append(list(tokens))
                print(json.dumps({"status": "ok", "ids": tokenized}), flush=True)

            elif command == "decode":
                ids_list = req.get("ids", [])
                decoded = []
                for ids in ids_list:
                    # decode
                    text = tokenizer.decode(ids)
                    decoded.append(text)
                print(json.dumps({"status": "ok", "texts": decoded}), flush=True)

            elif command == "get_special_tokens":
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id
                if pad_id is None:
                    pad_id = 0
                
                eos_ids = tokenizer.eos_token_id
                if eos_ids is None:
                    eos_ids = []
                elif isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                else:
                    eos_ids = list(eos_ids)

                print(json.dumps({
                    "status": "ok",
                    "pad_token_id": pad_id,
                    "eos_token_ids": eos_ids
                }), flush=True)
            else:
                print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}), flush=True)

        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)

if __name__ == "__main__":
    main()
