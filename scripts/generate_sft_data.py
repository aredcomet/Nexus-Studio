import os
import time
from tqdm import tqdm
from mlx_lm import load, generate
from dataclasses import dataclass
try:
    from .grpo_dataset import generate_math_problem
except ImportError:
    from grpo_dataset import generate_math_problem
import psycopg
from datetime import datetime

dsn = os.getenv("POSTGRES_GO_DSN")
if not dsn:
    raise ValueError("POSTGRES_GO_DSN environment variable is not set")
conn = psycopg.connect(dsn)
cur = conn.cursor()

MODEL_PATH = "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4"

@dataclass
class Problem:
    id: int
    problem: str
    answer: str
    problem_type: str
    solution: str | None = None
    time_to_solve: int | None = None
    created_at: datetime | None = None


def get_total_count():
    """
    Returns the total number of problems in the database.
    """
    cur.execute("SELECT count(*) FROM problemset")
    return cur.fetchone()[0]

def create_problemset(num_samples=2200):
    total_count = get_total_count()
    if total_count >= num_samples:
        print(f"Generated {total_count} problems already.")
        return
    for _ in tqdm(range(num_samples), desc="Generating Problem Set"):
        problem, answer, problem_type = generate_math_problem()
        cur.execute("INSERT INTO problemset (problem, answer, problem_type) VALUES (%s, %s, %s)", (problem, answer, problem_type))
        conn.commit()
    print(f"Generated {num_samples} problems")


def get_unsolved_problems(limit=200) -> list[Problem]:
    """Get problems from the database that don't have a solution yet."""
    cur.execute(
        "SELECT id, problem, answer, problem_type, solution, time_to_solve, created_at "
        "FROM problemset WHERE solution IS NULL ORDER BY id ASC LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    return [Problem(*row) for row in rows]

def save_batch_solutions(updates: list[tuple[str, int, int]]):
    """
    Perform a single batched UPDATE transaction for the whole batch.
    updates should be a list of (solution, time_to_solve, id) tuples.
    """
    cur.executemany(
        "UPDATE problemset SET solution = %s, time_to_solve = %s WHERE id = %s",
        updates
    )
    conn.commit()


from utils.mlx_batch_generator import MLXBatchGenerator

def batch_from_problems(problem_set, batch_size=4):
    """
    returns batches of the problem set
    """
    return [problem_set[x:x + batch_size] for x in range(0, len(problem_set), batch_size)]

def solve_problems(number=200):
    """
    Check database for unsolved problems and generate solutions for them in batches.
    """
    model, tokenizer = load(MODEL_PATH, tokenizer_config={"fix_mistral_regex": True})
    batch_generator = MLXBatchGenerator(model, tokenizer)
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    
    problems = get_unsolved_problems(limit=number)
    if not problems:
        print("No unsolved problems found.")
        return
        
    problem_batches = batch_from_problems(problems, batch_size=4)
    
    # Process problems in batches
    for batch in tqdm(problem_batches, desc="Solving Problem Batches"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_format.format(p.problem)}],
                add_generation_prompt=True
            )
            for p in batch
        ]
        
        start_time = time.time()
        # Use our custom batch generator with a realistic context window
        solutions = batch_generator.generate(
            prompts, 
            max_tokens=1024, 
            temperature=0.0, 
            verbose=False
        )
        batch_time = int(time.time() - start_time)
        time_per_problem = batch_time // len(batch)
        
        updates = []
        for problem, solution in zip(batch, solutions):
            clean_solution = solution.replace("[THINK]", "<think>").replace("[/THINK]", "</think>")
            updates.append((clean_solution, time_per_problem, problem.id))
            
        save_batch_solutions(updates)
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
    try:
        create_problemset(2200)
        solve_problems(2200)
    finally:
        cur.close()
        conn.close()
