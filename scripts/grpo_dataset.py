import re
import random

def generate_mixed_math_problem():
    """
    Generates a mix of 1-step and 2-step math problems.
    """
    a = random.randint(2, 12)
    b = random.randint(2, 12)
    c = random.randint(2, 12)
    
    # Mix of 1-step and 2-step problems to build a curriculum
    templates = [
        f"{a} + {b}",
        f"{a} * {b}",
        f"{a} - {b}",
        f"{a} + {b} * {c}",
        f"{a} * {b} + {c}",
        f"{a} * {b} - {c}",
        f"{a} - {b} + {c}"
    ]
    
    problem_str = random.choice(templates)
    exact_answer = eval(problem_str)
    
    display_str = problem_str.replace('*', ' × ')
    return display_str, str(exact_answer)

def generate_bodmas_problem():
    # 1. Generate random numbers
    a = random.randint(2, 10)
    b = random.randint(2, 12)
    c = random.randint(2, 6)
    d = random.randint(2, 5)
    
    # To ensure division results in a clean whole number, 
    # we make the numerator a perfect multiple of the denominator.
    divisor = random.randint(2, 5)
    multiplier = random.randint(2, 6)
    numerator = divisor * multiplier  # This guarantees 'numerator / divisor' is an integer
    
    # 2. Define different structural templates for the problems
    templates = [
        f"{a} + {b} * {c}",
        f"({a} + {b}) * ({c} - {d})",
        f"{numerator} / {divisor} + {a} * {b}",
        f"{a} * {b} - ({c} + {d})",
        f"{a} + {b} * ({numerator} / {divisor}) - {c}",
        f"({a} * {b}) + ({numerator} / {divisor})",
        f"{a} * {b} + {c}**2 - {d}"  # Includes an exponent (Order)
    ]
    
    # 3. Pick a random template
    problem_str = random.choice(templates)
    
    # 4. Safely calculate the exact answer using eval()
    # we cast to int if it ends in .0 just for cleaner display
    exact_answer = eval(problem_str)
    if isinstance(exact_answer, float) and exact_answer.is_integer():
        exact_answer = int(exact_answer)
        
    # Replace Python's syntax (*, /, **) with standard math symbols for display
    display_str = problem_str.replace('**', '^').replace('*', ' × ').replace('/', ' ÷ ')
    
    return display_str, str(exact_answer)

def generate_math_problem():
    """
    returns random problem from either generate_mixed_math_problem or generate_bodmas_problem
    """
    if random.random() < 0.5:
        p, a = generate_mixed_math_problem()
        return p, a, "mixed"
    else:
        p, a = generate_bodmas_problem()
        return p, a, "bodmas"

def generate_math_dataset(num_samples=100):
    dataset = []
    for _ in range(num_samples):
        question, ans, prob_type = generate_math_problem()
        
        prompt = (
            f"Solve the following math problem step-by-step. {prob_type} problem\n"
            "Ensure that your final numerical answer is the very last number in your response.\n\n"
            "Example:\n"
            "Problem: What is 2 + 3 * 4?\n"
            """Let's solve this step-by-step using the order of operations (PEMDAS/BODMAS: Parentheses/Brackets, Exponents/Orders, Multiplication and Division (from left to right), Addition and Subtraction (from left to right)).\n1. **Multiplication first**: 3 * 4 = 12\nThe expression now becomes: 2 + 12\n2. **Addition next**: 2 + 12 = 14\n**Final answer:** 14\n\n"""
            f"Problem: What is {question}?"
        )
        
        dataset.append({
            "prompt": prompt,
            "ground_truth": ans
        })
        
    return dataset

def reward_function(response: str, ground_truth: str) -> float:
    """
    Tagless reward function. Extracts the last number in the response.
    """
    reward = 0.0
    
    # Extract all numbers (including negative numbers) from the response
    numbers = re.findall(r'-?\d+', response)
    
    if not numbers:
        return 0.0
        
    # Check if the VERY LAST number matches the ground truth
    if numbers[-1] == ground_truth:
        reward += 2.0  # Perfect!
    elif ground_truth in numbers:
        reward += 0.5  # Partial credit: the right answer is in there somewhere!
        
    return reward

if __name__ == "__main__":
    q, a, prob_type = generate_math_problem()
    print(f"Sample ({prob_type}): {q} = {a}")
