import warnings
warnings.filterwarnings('ignore')

import os

class Config:
    def __init__(self):
        # =========================
        # Dataset / pipeline
        # =========================
        self.dataset_name_list = [
            'amc23', 'math500', 'minerva',
            'aime2025', 'aime2024',
            'olympiad',
        ]
        self.dataset_root_path = os.getenv("DATASET_ROOT", None)
        self.prompt_strategy = "Stepwise-Correction"

        # =========================
        # Backend selection
        # =========================
        # "openai" or "hf"
        self.backend_type = os.getenv("BACKEND_TYPE", "hf")

        # =========================
        # LLM model (either OpenAI or HF)
        # =========================
        # OpenAI일 때: 예) "gpt-4o"
        # HF일 때: 예) "meta-llama/Llama-3.1-8B-Instruct" / "google/gemma-3-12b-it"
        # self.backend_LLM = os.getenv("BACKEND_LLM", "google/gemma-3-27b-it")
        self.backend_LLM = os.getenv("BACKEND_LLM", "meta-llama/Llama-3.1-8B-Instruct")

        # -------- OpenAI params --------
        self.openai_LLM_base_url = os.getenv("OPENAI_BASE_URL", "xxx")
        self.openai_LLM_api_key  = os.getenv("OPENAI_API_KEY", "xxx")

        # =========================
        # Generation params (shared)
        # =========================
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.top_k = int(os.getenv("TOP_K", "0"))

        # 선택: 반복 방지/길이 패널티 등 필요하면 확장
        self.repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.0"))

        # =========================
        # Verification model (PRM)
        # =========================
        self.verification_model_cache_dir = os.getenv("VERIFIER_CACHE_DIR", os.getenv('CACHE_DIR'))
        self.verification_model = os.getenv("VERIFIER_MODEL", "UW-Madison-Lee-Lab/VersaPRM")

        # =========================
        # Results
        # =========================
        self.result_save_root_path = os.getenv("RESULT_SAVE_ROOT", "./outputs")

        # =========================
        # Stepwise correction control
        # =========================
        self.threshold = float(os.getenv("THRESHOLD", "0.5"))
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
