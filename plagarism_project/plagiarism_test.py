from transformers import AutoTokenizer, AutoModel
import torch
import ast

# ----------- CodeBERT Setup -----------
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

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


# ----------- AST Similarity -----------
def ast_similarity(code1, code2):
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
    except SyntaxError:
        return 0.0

    # ✅ Fixed here: _name_ instead of name
    nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
    nodes2 = [type(node).__name__ for node in ast.walk(tree2)]

    overlap = len(set(nodes1) & set(nodes2))
    total = len(set(nodes1) | set(nodes2))
    return overlap / total if total > 0 else 0


# ----------- Load Code Files -----------
with open("plagiarism_folder/file1.py", "r", encoding="utf-8") as f:
    code1 = f.read()
with open("plagiarism_folder/file2.py", "r", encoding="utf-8") as f:
    code2 = f.read()
    

# ----------- Run Both Models -----------
vec1 = embed_code(code1)
vec2 = embed_code(code2)

codebert_score = cosine_similarity(vec1, vec2)
ast_score = ast_similarity(code1, code2)

print("CodeBERT Similarity:", codebert_score)
print("AST Similarity:", ast_score)