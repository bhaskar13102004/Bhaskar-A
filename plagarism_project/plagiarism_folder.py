import os
import ast
import torch
from transformers import AutoTokenizer, AutoModel

# ===============================
# Load CodeBERT
# ===============================
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ===============================
# 1. AST-based similarity
# ===============================
def ast_similarity(code1, code2):
    try:
        tree1 = ast.dump(ast.parse(code1))
        tree2 = ast.dump(ast.parse(code2))
    except SyntaxError:
        return 0.0  # if file has errors, return 0
    set1, set2 = set(tree1.split()), set(tree2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 and set2 else 0.0

# ===============================
# 2. CodeBERT embedding similarity
# ===============================
def embed_code(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector.squeeze(0)

def codebert_similarity(code1, code2):
    vec1, vec2 = embed_code(code1), embed_code(code2)
    vec1, vec2 = vec1 / vec1.norm(), vec2 / vec2.norm()
    return float(torch.dot(vec1, vec2))

# ===============================
# 3. Compare all files in folder
# ===============================
folder_path = "plagiarism_folder"  # put your folder name here
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".py")]

for i in range(len(files)):
    for j in range(i + 1, len(files)):
        with open(files[i], "r", encoding="utf-8", errors="ignore") as f1, \
             open(files[j], "r", encoding="utf-8", errors="ignore") as f2:
            code1 = f1.read()
            code2 = f2.read()

        ast_score = ast_similarity(code1, code2)
        codebert_score = codebert_similarity(code1, code2)

        print(f"\nComparing {files[i]} and {files[j]}:")
        print(f"AST Similarity Score: {ast_score:.2f}")
        print(f"CodeBERT Similarity Score: {codebert_score:.2f}")

        if ast_score > 0.8 and codebert_score > 0.8:
            print("✅ Likely plagiarism (same logic and style)")
        elif codebert_score > 0.8 and ast_score < 0.5:
            print("⚠ Same style but different logic")
        else:
            print("❌ Not plagiarism")
            