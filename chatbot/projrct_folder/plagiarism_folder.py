import os
import ast
import torch
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT
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

# ---- CodeBERT FUNCTIONS ----
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
# Collect all Python files in the folder
files = [f for f in os.listdir() if f.endswith(".py") and f != "plagiarism_folder.py"]

# Compare each pair of files
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        with open(files[i], "r", encoding="utf-8", errors="ignore") as f1, \
     open(files[j], "r", encoding="utf-8", errors="ignore") as f2:
    code1 = f1.read()
    code2 = f2.read()34

        # AST score
        ast_score = ast_similarity(code1, code2)
        # CodeBERT score
        vec1 = embed_code(code1)
        vec2 = embed_code(code2)
        bert_score = cosine_similarity(vec1, vec2)

        print(f"\nComparing {files[i]} ↔ {files[j]}")
        print(f"  AST Score: {ast_score:.2f}")
        print(f"  CodeBERT Score: {bert_score:.2f}")

        if bert_score > 0.8 and ast_score > 0.7:
            print("  ✅ High chance of plagiarism!")
        elif bert_score > 0.8 and ast_score < 0.5:
            print("  ⚠ Same style but different logic.")
        else:
            print("  ❌ No strong plagiarism detected.")