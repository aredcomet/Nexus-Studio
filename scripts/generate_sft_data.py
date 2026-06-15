import json
import os
import sqlite3
import random
from tqdm import tqdm
from mlx_lm import load, generate, batch_generate
from dataclasses import dataclass
from grpo_dataset import generate_math_problem


conn = sqlite3.connect("problemset.db")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS problemset (id INTEGER PRIMARY KEY, problem TEXT, answer TEXT, type TEXT, solution TEXT)")
conn.commit()

@dataclass
class Problem:
    id: int
    question: str
    answer: str
    type: str
    solution: str | None = None

def create_problemset(num_samples=2200):
    for _ in tqdm(range(num_samples), desc="Generating Problem Set"):
        problem, answer, type = generate_math_problem()
        cur.execute("INSERT INTO problemset (problem, answer, type) VALUES (?, ?, ?)", (problem, answer, type))
        conn.commit()
    print(f"Generated {num_samples} problems")


def get_unsolved_problems(limit=200) -> list[Problem]:
    """Get problems from the database that don't have a solution yet."""
    cur.execute("SELECT * FROM problemset WHERE solution IS NULL ORDER BY id ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [Problem(*row) for row in rows]

def save_solution(id: int, solution: str):
    cur.execute("UPDATE problemset SET solution = ? WHERE id = ?", (solution, id))
    conn.commit()


def solve_problems(number=200):
    """
    Check database for unsolved problems and generate solutions for them
    """

    model, tokenizer = load("/Users/bran/.lmstudio/models/mlx-community/Ministral-3-8B-Reasoning-2512-4bit")
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    problems = get_unsolved_problems(limit=number)
    for problem in tqdm(problems):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_format.format(problem.question)}], add_generation_prompt=True)
        text = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=8192)
        save_solution(problem.id, text.replace("[THINK]", "<think>").replace("[/THINK]", "</think>"))


# def batch_from_problems(problem_set, batch_size=10):
#     """
#     returns batches of the problem set
#     """
    
#     return [problem_set[x:x + batch_size] for x in range(0, len(problem_set), batch_size)]


# def create_sft_dataset(output_dir="data/sft", train_size=2000, valid_size=200):
#     """
#     Generates a Supervised Fine-Tuning (SFT) dataset.
#     For SFT, we must provide both the Prompt AND the exact, perfect Answer.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    

#     prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
#     model, tokenizer = load("/Users/bran/.lmstudio/models/mlx-community/Ministral-3-8B-Reasoning-2512-4bit/")

#     def generate_jsonl(filename, num_samples):
#         filepath = os.path.join(output_dir, filename)
#         problem_set = [generate_math_problem()[0] for x in range(num_samples)]
#         problem_batches = batch_from_problems(problem_set, batch_size=10)
#         with open(filepath, 'w') as f:
#             for batch in tqdm(problem_batches, desc="Generating Batch responses"):

#                 prompts = [
#                     tokenizer.apply_chat_template([{"role": "user", "content": prompt_format.format(p)+"\n\nDo not use special formatting"}], add_generation_prompt=True)
#                     for p in batch
#                 ]

#                 result = batch_generate(model, tokenizer, prompts, verbose=False, return_prompt_caches=True, max_tokens=8192)
                
#                 # Write to file immediately so we don't lose data if it crashes!
#                 for p, solution in zip(batch, result.texts):                
#                     sample = {
#                         "messages": [
#                             {"role": "user", "content": prompt_format.format(p)},
#                             {"role": "assistant", "content": solution.replace("[THINK]", "<think>").replace("[/THINK]", "</think>")}
#                         ]
#                     }
#                     f.write(json.dumps(sample, ensure_ascii=False) + "\n")
#                     f.flush() # Ensure it actually writes to disk immediately
                
#         print(f"Generated {num_samples} samples in {filepath}")

#     generate_jsonl("train.jsonl", train_size)
#     generate_jsonl("valid.jsonl", valid_size)



if __name__ == "__main__":
    create_problemset(2200)
    solve_problems(2200)
    conn.close()
