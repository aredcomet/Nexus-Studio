import os
import json
import httpx
from tqdm import tqdm
try:
    from .grpo_dataset import generate_math_problem
except ImportError:
    from grpo_dataset import generate_math_problem
import psycopg
import time
from datetime import datetime
from dataclasses import dataclass

URL = "http://127.0.0.1:1234/api/v1/chat"
MODEL = "local/ministral-3-8b-reasoning-2512"
TIMEOUT = 300                                        # seconds

dsn = os.getenv("POSTGRES_GO_DSN")
if not dsn:
    raise ValueError("POSTGRES_GO_DSN environment variable is not set")
conn = psycopg.connect(dsn)
cur = conn.cursor()


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


def save_solution(id: int, solution: str, time_to_solve: int):
    """
    Perform a single UPDATE transaction.
    """
    cur.execute(
        "UPDATE problemset SET solution = %s, time_to_solve = %s WHERE id = %s",
        (solution, time_to_solve, id)
    )
    conn.commit()


def format_response(response: str) -> str:
    """
    Convert response content to raw output:
    <think>{reasoning}</think>{response}</s>

    - Format json strings double quotes un-escape characters
    eg: the user just said \"hello\" -> the user just said "hello"
    """
    text = json.loads(response)
    raw_output = ""
    for block in text["output"]:
        if block["type"] == "reasoning":
            raw_output += f"<think>{block['content']}</think>"
        elif block["type"] == "message":
            raw_output += f"{block['content']}"
    raw_output += "</s>"
    
    return raw_output


def get_solution():
    problems = get_unsolved_problems(limit=2200)
    if not problems:
        print("No unsolved problems found.")
        return
    
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    headers = {"Content-Type": "application/json"}
    
    # Create a list of lists of 5 items
    problemset = [problems[x:x + 5] for x in range(0, len(problems), 5)]
    print(f"Generating solutions for {len(problemset)} batches\nAfter each batch, wait 60 seconds to cool down")
    
    for batch in tqdm(problemset, desc="Solving batches"):
        for problem in batch:
            max_retries = 3
            retry_delay = 5  # seconds
            success = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    r = httpx.post(
                        URL,
                        headers=headers,
                        data=json.dumps({
                            "model": MODEL,
                            "input": prompt_format.format(problem.problem)
                        }),
                        timeout=TIMEOUT,
                    )
                    r.raise_for_status()
                    data = r.text
                    solution = format_response(data)
                    save_solution(problem.id, solution, r.elapsed.seconds)
                    success = True
                    break
                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                    print(f"\n[Attempt {attempt}/{max_retries}] Error solving problem ID {problem.id}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
            
            if not success:
                print(f"\n[FAILED] Could not solve problem ID {problem.id} after {max_retries} attempts. Skipping.")
                
        # Cool down after each batch of 5
        time.sleep(60)


if __name__ == "__main__":
    get_solution()
