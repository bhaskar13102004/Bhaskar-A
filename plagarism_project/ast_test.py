import ast

# Function to convert code into AST string
def normalize_code(code):
    tree = ast.parse(code)   # Parse code into AST
    return ast.dump(tree)    # Dump AST structure as string

# Function to calculate similarity (very simple: how many chars match)
def simple_similarity(code1, code2):
    ast1 = normalize_code(code1)
    ast2 = normalize_code(code2)
    # Compare character overlap between AST strings
    matches = sum(1 for a, b in zip(ast1, ast2) if a == b)
    return matches / max(len(ast1), len(ast2))

# Example codes
code1 = "def add(a, b): return a + b"
code2 = "def multiply(a, b): return a * b"

print("AST Similarity:", simple_similarity(code1, code2))
# Read code from two files
with open("file1.py", "r") as f:
    code1 = f.read()

with open("file2.py", "r") as f:
    code2 = f.read()

print("AST Similarity:", simple_similarity(code1, code2))