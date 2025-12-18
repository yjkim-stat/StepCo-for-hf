import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

from config import Config


## load verification model
config = Config()
good_token = '+'
bad_token = '-'

if config.verification_model in ['UW-Madison-Lee-Lab/VersaPRM']:
    step_tag = ' \n\n\n\n'
elif config.verification_model in ['peiyi9979/math-shepherd-mistral-7b-prm']:
    step_tag = 'ки'
else:
    raise KeyError()

device = torch.device("cuda")

if config.verification_model in ['UW-Madison-Lee-Lab/VersaPRM']:
    plus_tag_id = self.tokenizer.encode(positive_tag)[-1]
    minus_tag_id = self.tokenizer.encode(negative_tag)[-1]
    candidate_tokens = [plus_tag_id, minus_tag_id]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1] 
elif config.verification_model in ['peiyi9979/math-shepherd-mistral-7b-prm']:
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902

tokenizer = AutoTokenizer.from_pretrained(config.verification_model, token=os.getenv('HF_TOKEN'), cache_dir=os.getenv('CACHE_DIR'),)
model = AutoModelForCausalLM.from_pretrained(
    config.verification_model, 
    torch_dtype=torch.float16, 
    trust_remote_code=True, 
    quantization_config=quantization_config, 
    cache_dir=os.getenv('CACHE_DIR'),
    attn_implementation=os.getenv('ATTN_IMPLEMENTATION', "flash_attention_2"),
    token=os.getenv('HF_TOKEN'),
    local_files_only=True,
    use_cache=True).eval()

def step_verify_score(input_seq):
    if config.verification_model == 'UW-Madison-Lee-Lab/VersaPRM':
        return step_verify_score_for_versaprm(input_seq)
    elif config.verification_model == 'peiyi9979/math-shepherd-mistral-7b-prm':
        return step_verify_score_for_msm7b(input_seq)
    else:
        raise KeyError()

def step_verify_score_for_msm7b(input_seq):
    input_id = torch.tensor([tokenizer.encode(input_seq)]).to(device)
    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_id == step_tag_id]

    return step_scores.cpu().detach().numpy().tolist()


def step_verify_score_for_versaprm(input_seq: str):
    """
    Args:
        input_seq: 
            e.g. "Q: ... \n A: step1 ки\n step2 ки\n ..."
    Returns:
        List[float]: step-wise correctness probabilities (P(good))
    """

    # tokenize
    input_ids = tokenizer.encode(input_seq, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab]

        # slice to [+ , -]
        logits = logits[:, :, candidate_tokens]          # [1, seq_len, 2]
        probs = torch.softmax(logits, dim=-1)[:, :, 0]   # P(good)

        # select positions where step_tag occurs
        mask = (input_ids == step_tag_id)
        step_scores = probs[mask]

    # safety check
    if step_scores.numel() == 0:
        raise ValueError(
            f"No step_tag '{step_tag}' found in input_seq. "
            f"Cannot compute step-level verification."
        )

    return step_scores.cpu().tolist()
