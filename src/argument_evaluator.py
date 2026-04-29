import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import LongformerTokenizer
from models.longform import build_model


app = FastAPI()

# Load model once at startup
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = build_model()
model.eval()

MAX_LENGTH = 2048


class CompareRequest(BaseModel):
    arg_a: str
    arg_b: str
    op_text: str = ""


def run_comparison(arg_a: str, arg_b: str, op_text: str = "") -> dict:
    """
    Triple-encode (OP, ARG_A, ARG_B) as a single sequence and return
    head-to-head probabilities.

    Label 0 = ARG_A wins, Label 1 = ARG_B wins.
    """
    # Match training format: tokenizer(op, arg_a + " </s> " + arg_b)
    combined_b = arg_a + " </s> " + arg_b

    inputs = tokenizer(
        op_text,
        combined_b,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    # Global attention on [CLS] token only, matching training setup
    global_attention_mask = torch.zeros_like(inputs["input_ids"])
    global_attention_mask[:, 0] = 1
    inputs["global_attention_mask"] = global_attention_mask

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=inputs["global_attention_mask"],
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    p_a = round(probs[0].item() * 100, 1)   # P(A wins)
    p_b = round(probs[1].item() * 100, 1)   # P(B wins)
    winner = "A" if p_a >= p_b else "B"
    confidence = max(p_a, p_b)

    return {
        "winner": winner,
        "confidence": confidence,
        "score_a": p_a,
        "score_b": p_b,
    }


@app.post("/compare")
def compare(request: CompareRequest):
    """
    Compare two arguments head-to-head.

    For robustness, run twice with positions swapped and average the
    probabilities — this cancels any residual positional bias.
    """
    forward  = run_comparison(request.arg_a, request.arg_b, request.op_text)
    backward = run_comparison(request.arg_b, request.arg_a, request.op_text)

    # In the backward pass, A and B are swapped:
    # backward["score_a"] is P(arg_b wins) and backward["score_b"] is P(arg_a wins)
    avg_score_a = round((forward["score_a"] + backward["score_b"]) / 2, 1)
    avg_score_b = round((forward["score_b"] + backward["score_a"]) / 2, 1)

    winner = "A" if avg_score_a >= avg_score_b else "B"
    confidence = max(avg_score_a, avg_score_b)

    return {
        "winner": winner,
        "confidence": confidence,
        "score_a": avg_score_a,
        "score_b": avg_score_b,
    }


# Serve frontend
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
def root():
    return FileResponse("src/static/index.html")
