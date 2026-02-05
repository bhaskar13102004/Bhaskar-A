import ast
import torch
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT model
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ---- AST FUNCTIONS ----
def normalize_code(code):
    tree = ast.parse(code)
    return ast.dump(tree)

def ast_similarity(code1, code2):
    ast1 = normalize_code(code1)
    ast2 = normalize_code(code2)
    matches = sum(1 for a, b in zip(ast1, ast2) if a == b)
    return matches / max(len(ast1), len(ast2))

# ---- CODEBERT FUNCTIONS ----
def embed_code(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector.squeeze(0)

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / vec1.norm()
    vec2 = vec2 / vec2.norm()
    return float(torch.dot(vec1, vec2))

# ---- MAIN PROGRAM ----
with open("file1.py", "r") as f:
    code1 = f.read()

with open("file2.py", "r") as f:
    code2 = f.read()

# Run both detectors
ast_score = ast_similarity(code1, code2)
vec1 = embed_code(code1)
vec2 = embed_code(code2)
bert_score = cosine_similarity(vec1, vec2)

print(f"AST Similarity Score: {ast_score:.2f}")
print(f"CodeBERT Similarity Score: {bert_score:.2f}")

# Combined judgment
if bert_score > 0.8 and ast_score > 0.7:
    print("✅ High chance of plagiarism!")
elif bert_score > 0.8 and ast_score < 0.5:
    print("⚠ Same style but different logic.")
else:
    print("❌ No strong plagiarism detected.")