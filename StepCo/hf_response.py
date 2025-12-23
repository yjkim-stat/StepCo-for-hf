from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
# from transformers import Gemma3ForConditionalGeneration

from config import Config


config = Config()

_HF = {"tokenizer": None, "model": None, "device": None}

def _load_hf_once(model_name: str, cache_dir: str | None = None):
    """
    Load HF model/tokenizer ONCE and cache them.
    Supports llama & gemma families with 4-bit quantization.
    """
    if _HF["model"] is not None:
        return

    cache_dir = cache_dir or os.getenv("CACHE_DIR")
    attn_impl = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")
    hf_token = os.getenv("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None

    # ---- BitsAndBytes 4bit config ----
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        use_fast=True,
    )

    processor = None

    # ===============================
    # LLaMA family
    # ===============================
    if "llama" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            attn_implementation=attn_impl,
            device_map=device_map,
            token=hf_token,
        )

        # eos / pad 안전 세팅
        eos_id = tokenizer.eos_token_id or model.config.eos_token_id
        pad_id = tokenizer.pad_token_id

        if pad_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = eos_id

        model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = pad_id

        meta = {
            "family": "llama",
            "is_multimodal": False,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

    # ===============================
    # Gemma family
    # ===============================
    elif "gemma" in model_name.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            attn_implementation=attn_impl,
            device_map=device_map,
            token=hf_token,
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token,
        )

        eos_id = tokenizer.eos_token_id or model.config.eos_token_id
        pad_id = tokenizer.pad_token_id or 0

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.decode(pad_id)

        model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = pad_id

        meta = {
            "family": "gemma",
            "is_multimodal": True,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

    else:
        raise KeyError(f"Unsupported model family: {model_name}")

    model.eval()

    _HF["model"] = model
    _HF["tokenizer"] = tokenizer
    _HF["processor"] = processor
    _HF["meta"] = meta


def answered_by_hf(user_input: str) -> str:
    _load_hf_once(
        model_name=config.backend_LLM,
        cache_dir=getattr(config, "hf_cache_dir", None),
    )

    model = _HF["model"]
    tokenizer = _HF["tokenizer"]
    meta = _HF["meta"]

    # Chat template 있으면 사용
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_input

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=getattr(config, "top_k", 0),
        pad_token_id=meta["pad_token_id"],
        eos_token_id=meta["eos_token_id"],
    )

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()

    if text == "":
        raise ValueError("Empty string from HF model")

    return text