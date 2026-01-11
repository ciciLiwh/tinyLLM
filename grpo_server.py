import json
import uuid
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tokenizers import ByteLevelBPETokenizer

from model import DecoderOnlyTransformer, select_device


VOCAB_PATH = Path("data") / "bbpe" / "vocab.json"
MERGES_PATH = Path("data") / "bbpe" / "merges.txt"
CKPT_PATH = Path("out") / "grpo_epoch1.pt"
MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.9
TOP_K = 50  # set to None to disable


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    message: str = Field(..., description="User message")
    max_new_tokens: int = Field(MAX_NEW_TOKENS, description="Max tokens to generate")
    temperature: float = Field(TEMPERATURE, description="Sampling temperature")
    top_k: int = Field(TOP_K, description="Top-k filter (use 0/None to disable)")


class NewConversationResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, str]]


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    messages: List[Dict[str, str]]


class DeleteMessageResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, str]]


def load_model(ckpt_path: Path, device: torch.device) -> DecoderOnlyTransformer:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train GRPO first.")
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    cfg = checkpoint["config"]
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        num_layers=cfg["num_layers"],
        num_q_heads=cfg["num_q_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        moe_hidden=cfg["moe_hidden"],
        num_experts=cfg["num_experts"],
        max_seq_len=cfg["max_seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def build_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}:\n{content}")
    lines.append("Assistant:\n")
    return "\n".join(lines)


def sample_response(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPETokenizer,
    messages: List[Dict[str, str]],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_seq_len: int,
) -> str:
    prompt = build_prompt(messages)
    ids = tokenizer.encode(prompt).ids
    ids = ids[-(max_seq_len - 1):]
    for _ in range(max_new_tokens):
        x = torch.tensor(ids[-max_seq_len:], dtype=torch.long, device=device)[None, ...]
        logits = model(x)[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return tokenizer.decode(ids)


device = select_device()
tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
model = load_model(CKPT_PATH, device)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store: {id: [{"role": "user"/"assistant", "content": str}, ...]}
CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}


@app.post("/conversation", response_model=NewConversationResponse)
def new_conversation() -> NewConversationResponse:
    convo_id = uuid.uuid4().hex
    CONVERSATIONS[convo_id] = []
    return NewConversationResponse(conversation_id=convo_id, messages=CONVERSATIONS[convo_id])


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    history = CONVERSATIONS[req.conversation_id]
    history.append({"role": "user", "content": req.message})
    reply_text = sample_response(
        model=model,
        tokenizer=tokenizer,
        messages=history,
        device=device,
        max_new_tokens=min(req.max_new_tokens, MAX_NEW_TOKENS),
        temperature=req.temperature,
        top_k=req.top_k,
        max_seq_len=MAX_SEQ_LEN,
    )
    reply_only = reply_text.split("Assistant:\n")[-1].strip()
    history.append({"role": "assistant", "content": reply_only})
    return ChatResponse(conversation_id=req.conversation_id, reply=reply_only, messages=history)


@app.delete("/conversation/{conversation_id}/message/{index}", response_model=DeleteMessageResponse)
def delete_message(conversation_id: str, index: int) -> DeleteMessageResponse:
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    history = CONVERSATIONS[conversation_id]
    if index < 0 or index >= len(history):
        raise HTTPException(status_code=400, detail="index out of range")
    history.pop(index)
    return DeleteMessageResponse(conversation_id=conversation_id, messages=history)


@app.delete("/conversation/{conversation_id}", response_model=NewConversationResponse)
def delete_conversation(conversation_id: str) -> NewConversationResponse:
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    CONVERSATIONS.pop(conversation_id, None)
    convo_id = uuid.uuid4().hex
    CONVERSATIONS[convo_id] = []
    return NewConversationResponse(conversation_id=convo_id, messages=[])


@app.get("/conversation/{conversation_id}", response_model=NewConversationResponse)
def get_conversation(conversation_id: str) -> NewConversationResponse:
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    return NewConversationResponse(conversation_id=conversation_id, messages=CONVERSATIONS[conversation_id])
