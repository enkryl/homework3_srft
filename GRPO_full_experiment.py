"""
GRPO_full_experiment.py — Полный экспериментальный пайплайн для Target24.

3 метода обучения:
  Step 1: GRPO-only (reward-only, curriculum + mixing)
  Step 2: SFT → GRPO (gold trajectories → reward)
  Step 3: SRFT (SFT warmup + GRPO)

Особенности:
  - vLLM для ускорения генерации (training + eval)
  - Curriculum learning (от простого к сложному)
  - Mixing прошлых уровней при обучении
  - Подробная pass@k оценка с диагностикой по задачам
  - Все логи в JSONL
"""

# ============================================================
# 0) SETUP
# ============================================================
import os, sys, re, json, time, random, gc, logging
from pathlib import Path
from collections import Counter

import unsloth
from unsloth import FastLanguageModel

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainerCallback

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from base.data import Data
from envs.target24.env import Target24Env, LEVEL_CONFIG, MAX_LEVEL
from envs.target24.verifier import Target24Verifier

env = Target24Env()

# ============================================================
# 1) CONFIG
# ============================================================
MODEL_NAME   = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN  = 512
LORA_RANK    = 16
SEED         = 42

DATA_DIR     = Path("data_v2")

# Логи и адаптеры — можно менять перед каждым методом
# (logs_grpo_only уже скачаны, теперь запускаем sft_grpo)
LOGS_DIR     = Path("logs_sft_grpo");  LOGS_DIR.mkdir(exist_ok=True)
RUNS_DIR     = Path("runs_sft_grpo");  RUNS_DIR.mkdir(exist_ok=True)

# Hard levels (определены статически)
HARD_LEVELS  = [8, 9, 10]

# System prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant. You always first think about the "
    "reasoning process in the mind and then provides the user with "
    "the answer.\n"
    "The reasoning process and answer are enclosed within "
    "'<think>' '</think>' and '<answer>' '</answer>' tags, "
    "respectively, e.g.,\n"
    "<think>\nA detailed reasoning process here, with possible reflections "
    "including but not limited to reviewing previous steps for errors, "
    "exploring alternative approaches, and considering possible refinements.\n"
    "</think>\n<answer>\nReply to user here.\n</answer>. "
    "Please reason step by step, and put your final answer within "
    "<answer> </answer> tags."
)

# Generation params for eval
GEN_TEMPERATURE  = 0.7
GEN_TOP_P        = 0.95
GEN_MAX_TOKENS   = 512
N_SAMPLES_PASSK  = 128
EVAL_LIMIT       = 50

# GRPO training params
PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 1
NUM_GENERATIONS  = 2
MAX_NEW_TOKENS   = 96

USE_VLLM         = False
VLLM_GPU_UTIL    = 0.85

# Curriculum: steps per level
STEPS_PER_LEVEL = {
    1: 80,  2: 100, 3: 120, 4: 150, 5: 180,
    6: 150, 7: 180, 8: 200, 9: 220, 10: 250,
}

random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 2) LOAD MODEL + LoRA
# ============================================================
from unsloth import FastLanguageModel

def load_model_lora():
    """Загружает модель и применяет LoRA. fast_inference для vLLM."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=16, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer

# ============================================================
# 3) DATASET LOADER
# ============================================================
def get_dataset_for_level(split: str, level: int):
    """Загружает dataset для одного уровня из JSONL файла."""
    path = DATA_DIR / f"{split}_L{level}.jsonl"
    if not path.exists():
        path = DATA_DIR / f"{split}.jsonl"
    ds = load_dataset("json", data_files=str(path), split="train")
    if path.name in (f"{split}.jsonl",):
        ds = ds.filter(lambda x: int(x.get("difficulty", 1)) == level)
    ds = ds.shuffle(seed=SEED)
    def map_fn(x):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": x["question"]},
            ],
            "answer": x.get("answer", ""),
            "metadata": x.get("metadata", None),
        }
    return ds.map(map_fn)

def get_sft_dataset():
    """Загружает SFT dataset с gold trajectories."""
    path = DATA_DIR / "sft_gold.jsonl"
    ds = load_dataset("json", data_files=str(path), split="train")
    ds = ds.shuffle(seed=SEED)
    return ds

# ============================================================
# 4) REWARD FUNCTIONS
# ============================================================
ANSWER_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.I)
THINK_BLOCK  = re.compile(r"<think>\s*.*?\s*</think>",    re.DOTALL | re.I)

def _extract_expr(text: str) -> str:
    m = ANSWER_BLOCK.search(text or "")
    if not m: return ""
    expr = (m.group(1) or "").strip()
    expr = expr.split("==",1)[0].split("=",1)[0].strip()
    return expr

def format_reward_func(completions, **kw):
    """Reward за правильный формат ответа."""
    out = []
    for c in completions:
        t = c[0]["content"]; r = 0.0
        if THINK_BLOCK.search(t):  r += 0.05
        if ANSWER_BLOCK.search(t): r += 0.10
        low = (t or "").strip().lower()
        if low.startswith("<think>") and low.endswith("</answer>"): r += 0.05
        out.append(r)
    return out

def validity_reward_func(prompts, completions, metadata, **kw):
    """Reward если выражение синтаксически и rule-valid."""
    out = []
    for c, m in zip(completions, metadata):
        expr = _extract_expr(c[0]["content"])
        if not expr or not m:
            out.append(0.0); continue
        numbers = m.get("numbers", [])
        ops = {op for op in str(m.get("allowed_ops","+ - * /")).split() if op in {"+","-","*","/"}}
        parens = bool(m.get("allow_parentheses", True))
        val = env.verifier.try_eval_expression(expr, numbers, ops, parens)
        out.append(0.25 if val is not None else 0.0)
    return out

def distance_reward_func(prompts, completions, answer, metadata, **kw):
    """Dense reward: меньше |val - target| → больше reward."""
    out = []
    for c, a, m in zip(completions, answer, metadata):
        expr = _extract_expr(c[0]["content"])
        if not expr or not m:
            out.append(0.0); continue
        try: target = int(str(a).strip())
        except: out.append(0.0); continue
        numbers = m.get("numbers", [])
        ops = {op for op in str(m.get("allowed_ops","+ - * /")).split() if op in {"+","-","*","/"}}
        parens = bool(m.get("allow_parentheses", True))
        val = env.verifier.try_eval_expression(expr, numbers, ops, parens)
        if val is None: out.append(0.0); continue
        diff = abs(val - target)
        out.append(0.30 / (1.0 + diff))
    return out

def correctness_reward_func(prompts, completions, answer, metadata, **kw):
    """Sparse reward: 1.0 если verifier подтверждает."""
    out = []
    for a, m, c in zip(answer, metadata, completions):
        d = Data(question="", answer=a, difficulty=1, metadata=m)
        out.append(1.0 if env.verify(d, c[0]["content"]) else 0.0)
    return out

reward_funcs = [format_reward_func, validity_reward_func,
                distance_reward_func, correctness_reward_func]

# ============================================================
# 5) LOGGING
# ============================================================
def make_logger(name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []; logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

class JSONLCallback(TrainerCallback):
    """Записывает метрики в JSONL."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        open(self.path, "w").close()
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        rec = {"step": int(state.global_step)}
        rec.update({k: float(v) if isinstance(v,(int,float)) else v
                     for k,v in logs.items()})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ============================================================
# 6) EVAL (pass@k с vLLM + диагностика)
# ============================================================
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

@torch.inference_mode()
def generate_samples_hf(model, tokenizer, question, n_samples=128):
    """Генерация через HuggingFace (fallback если нет vLLM)."""
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    bs = min(16, n_samples)
    for s in range(0, n_samples, bs):
        nb = min(bs, n_samples - s)
        bi = {k: v.expand(nb,-1) for k,v in inputs.items()}
        out = model.generate(**bi, max_new_tokens=GEN_MAX_TOKENS,
                             do_sample=True, temperature=GEN_TEMPERATURE,
                             top_p=GEN_TOP_P)
        for seq in out:
            completions.append(tokenizer.decode(seq[prompt_len:],
                                                 skip_special_tokens=True))
    return completions[:n_samples]


def generate_samples_vllm(model, tokenizer, question, n_samples=128):
    """Генерация через vLLM (быстрее в 5-10x)."""
    from vllm import SamplingParams

    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    sampling = SamplingParams(
        temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS, n=n_samples,
    )
    outputs = model.fast_generate(
        [text], sampling_params=sampling, lora_request=None,
    )
    # fast_generate возвращает list[RequestOutput]
    completions = []
    for out in outputs:
        for o in out.outputs:
            completions.append(o.text)
    return completions[:n_samples]


def eval_pass_at_k(model, tokenizer, level, limit=EVAL_LIMIT,
                   k_values=None, tag="model", use_vllm=False,
                   save_diagnostics=True):
    """
    Оценивает pass@k на val для данного уровня.
    Сохраняет подробную диагностику: какие задачи решены, какие нет.
    """
    if k_values is None:
        k_values = [1, 4, 8, 16, 32, 64, 128]

    val_path = DATA_DIR / f"val_L{level}.jsonl"
    items = Data.from_jsonl_file(str(val_path))
    subset = items[:limit] if limit else items

    gen_fn = generate_samples_vllm if use_vllm else generate_samples_hf
    per_task = []

    for i, d in enumerate(subset):
        comps = gen_fn(model, tokenizer, d.question, N_SAMPLES_PASSK)
        nc = sum(1 for c in comps if env.verify(d, c))
        per_task.append({
            "idx": i,
            "n_correct": nc,
            "n_total": N_SAMPLES_PASSK,
            "numbers": d.metadata.get("numbers"),
            "target": d.metadata.get("target"),
            "gold_expr": d.metadata.get("gold_expr"),
            "solved": nc > 0,  # хотя бы 1 correct
        })
        if (i+1) % 10 == 0:
            solved_so_far = sum(1 for t in per_task if t["solved"])
            print(f"  {tag} L{level} [{i+1}/{len(subset)}] "
                  f"this={nc}/{N_SAMPLES_PASSK} "
                  f"solved={solved_so_far}/{len(per_task)}")

    # pass@k stats
    pk = {}
    for k in k_values:
        if k > N_SAMPLES_PASSK: continue
        pk[k] = float(np.mean([pass_at_k(t["n_total"],t["n_correct"],k)
                                for t in per_task]))

    result = {"level": level, "pass_at_k": pk, "per_task": per_task, "tag": tag}

    # Подробная диагностика
    solved = [t for t in per_task if t["solved"]]
    unsolved = [t for t in per_task if not t["solved"]]

    print(f"\n  L{level} SUMMARY: {len(solved)}/{len(per_task)} solved "
          f"(pass@128={pk.get(128,0):.4f})")
    if unsolved:
        print(f"  Unsolved tasks ({len(unsolved)}):")
        for t in unsolved[:10]:  # показываем первые 10
            print(f"    #{t['idx']}: numbers={t['numbers']} "
                  f"target={t['target']} gold={t['gold_expr']}")
        if len(unsolved) > 10:
            print(f"    ... и ещё {len(unsolved)-10}")

    # Сохраняем диагностику
    if save_diagnostics:
        diag_path = LOGS_DIR / f"{tag}_diag_L{level}.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result

# ============================================================
# 7) STEP 1: GRPO-ONLY TRAINING
# ============================================================
from trl import GRPOConfig, GRPOTrainer

def train_grpo_curriculum(model, tokenizer, tag="grpo_only",
                          levels=None, mix_previous=True):
    """
    GRPO с curriculum learning + vLLM ускорение.
    mix_previous=True: на каждом уровне смешиваем данные всех предыдущих.
    """
    if levels is None:
        levels = list(range(1, MAX_LEVEL + 1))

    all_metrics = []
    for lvl in levels:
        stage_dir = RUNS_DIR / f"{tag}_L{lvl}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{tag}_L{lvl}.log"
        jsonl_log = LOGS_DIR / f"{tag}_metrics_L{lvl}.jsonl"
        logger = make_logger(f"{tag}_L{lvl}", log_file)

        logger.info(f"=== {tag} LEVEL {lvl} ===")

        # Dataset: mix previous levels
        if mix_previous and lvl > min(levels):
            dss = [get_dataset_for_level("train", k) for k in levels if k <= lvl]
            ds = concatenate_datasets(dss).shuffle(seed=SEED)
            logger.info(f"Mixed dataset: {len(ds)} (levels {min(levels)}..{lvl})")
        else:
            ds = get_dataset_for_level("train", lvl)
            logger.info(f"Dataset: {len(ds)} (level {lvl})")

        args = GRPOConfig(
            output_dir=str(stage_dir), seed=SEED,
            use_vllm=USE_VLLM,
            vllm_gpu_memory_utilization=VLLM_GPU_UTIL if USE_VLLM else 0.9,
            learning_rate=5e-6, optim="adamw_8bit",
            lr_scheduler_type="cosine", warmup_ratio=0.05,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_generations=NUM_GENERATIONS,
            max_prompt_length=256, max_completion_length=MAX_NEW_TOKENS,
            max_steps=STEPS_PER_LEVEL[lvl],
            logging_steps=10, save_steps=0, report_to=[],
        )
        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer,
            args=args, train_dataset=ds, reward_funcs=reward_funcs,
        )
        trainer.add_callback(JSONLCallback(jsonl_log))

        t0 = time.time()
        trainer.train()
        dt = time.time() - t0
        logger.info(f"Train done in {dt/60:.1f} min")

        # Save adapter
        adapter_dir = stage_dir / "lora_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"Saved adapter: {adapter_dir}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        all_metrics.append({"level": lvl, "train_time_min": dt/60,
                            "steps": STEPS_PER_LEVEL[lvl]})

    # Save final adapter
    final = RUNS_DIR / f"{tag}_final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"  {tag} final LoRA saved: {final}")
    return all_metrics

# ============================================================
# 8) STEP 2: SFT → GRPO
# ============================================================
from trl import SFTConfig, SFTTrainer

def train_sft(model, tokenizer, tag="sft"):
    """SFT на gold trajectories."""
    sft_ds = get_sft_dataset()
    stage_dir = RUNS_DIR / tag
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{tag}.log"
    jsonl_log = LOGS_DIR / f"{tag}_metrics.jsonl"
    logger = make_logger(tag, log_file)

    logger.info(f"SFT dataset: {len(sft_ds)} examples")

    # Преобразуем messages → text (чтобы не мучиться с formatting_func)
    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    sft_ds = sft_ds.map(to_text)

    args = SFTConfig(
        output_dir=str(stage_dir), seed=SEED,
        dataset_text_field="text",
        learning_rate=2e-5, optim="adamw_8bit",
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1, max_seq_length=MAX_SEQ_LEN,
        logging_steps=10, save_steps=0, report_to=[],
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        args=args, train_dataset=sft_ds,
    )
    trainer.add_callback(JSONLCallback(jsonl_log))

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    logger.info(f"SFT done in {dt/60:.1f} min")

    adapter_dir = stage_dir / "lora_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"Saved SFT adapter: {adapter_dir}")
    return {"train_time_min": dt/60}

# ============================================================
# 9) STEP 3: SRFT
# ============================================================
def train_srft(model, tokenizer, tag="srft", levels=None, mix_previous=True):
    """
    SRFT: SFT warmup (1 epoch) + GRPO curriculum (сокращённый).
    """
    if levels is None:
        levels = list(range(1, MAX_LEVEL + 1))

    sft_ds = get_sft_dataset()

    # Преобразуем messages → text
    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    sft_ds = sft_ds.map(to_text)

    # SFT warmup
    print("SRFT: SFT warmup...")
    sft_dir = RUNS_DIR / f"{tag}_sft_warmup"
    sft_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(sft_dir), seed=SEED,
        dataset_text_field="text",
        learning_rate=2e-5, optim="adamw_8bit",
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1, max_seq_length=MAX_SEQ_LEN,
        logging_steps=10, save_steps=0, report_to=[],
    )
    sft_trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        args=sft_args, train_dataset=sft_ds,
    )
    sft_log = LOGS_DIR / f"{tag}_sft_warmup_metrics.jsonl"
    sft_trainer.add_callback(JSONLCallback(sft_log))
    sft_trainer.train()
    print("SRFT: SFT warmup done")

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # GRPO с curriculum (сокращённые шаги)
    srft_steps = {k: max(100, v // 2) for k, v in STEPS_PER_LEVEL.items()}

    all_metrics = []
    for lvl in levels:
        stage_dir = RUNS_DIR / f"{tag}_L{lvl}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{tag}_L{lvl}.log"
        jsonl_log = LOGS_DIR / f"{tag}_metrics_L{lvl}.jsonl"
        logger = make_logger(f"{tag}_L{lvl}", log_file)
        logger.info(f"=== SRFT LEVEL {lvl} ===")

        if mix_previous and lvl > min(levels):
            dss = [get_dataset_for_level("train", k) for k in levels if k <= lvl]
            ds = concatenate_datasets(dss).shuffle(seed=SEED)
        else:
            ds = get_dataset_for_level("train", lvl)

        args = GRPOConfig(
            output_dir=str(stage_dir), seed=SEED,
            use_vllm=USE_VLLM,
            vllm_gpu_memory_utilization=VLLM_GPU_UTIL if USE_VLLM else 0.9,
            learning_rate=3e-6, optim="adamw_8bit",
            lr_scheduler_type="cosine", warmup_ratio=0.05,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_generations=NUM_GENERATIONS,
            max_prompt_length=256, max_completion_length=MAX_NEW_TOKENS,
            max_steps=srft_steps[lvl],
            logging_steps=10, save_steps=0, report_to=[],
        )
        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer,
            args=args, train_dataset=ds, reward_funcs=reward_funcs,
        )
        trainer.add_callback(JSONLCallback(jsonl_log))
        t0 = time.time()
        trainer.train()
        dt = time.time() - t0
        logger.info(f"Train done in {dt/60:.1f} min")

        adapter_dir = stage_dir / "lora_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        all_metrics.append({"level":lvl, "train_time_min":dt/60})

    final = RUNS_DIR / f"{tag}_final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"  {tag} final LoRA saved: {final}")
    return all_metrics

# ============================================================
# 10) FULL EXPERIMENT RUNNER
# ============================================================
def run_full_experiment():
    """Запускает все 3 метода + eval + анализ."""

    all_results = {}

    # --- Step 1: GRPO-only ---
    print("\n" + "="*60)
    print("STEP 1: GRPO-ONLY (curriculum + mixing)")
    print("="*60)
    model, tokenizer = load_model_lora()
    train_grpo_curriculum(model, tokenizer, tag="grpo_only", mix_previous=True)

    FastLanguageModel.for_inference(model)
    print("\nEvaluating GRPO-only...")
    grpo_results = {}
    for lvl in range(1, MAX_LEVEL + 1):
        r = eval_pass_at_k(model, tokenizer, lvl, tag="grpo_only")
        grpo_results[lvl] = r
        print(f"  L{lvl} pass@1={r['pass_at_k'].get(1,0):.4f} "
              f"pass@128={r['pass_at_k'].get(128,0):.4f}")
    all_results["grpo_only"] = grpo_results

    with open(LOGS_DIR / "grpo_only_pass_at_k.json", "w") as f:
        json.dump({str(k): v["pass_at_k"] for k,v in grpo_results.items()}, f, indent=2)

    del model, tokenizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Step 2: SFT → GRPO ---
    print("\n" + "="*60)
    print("STEP 2: SFT -> GRPO")
    print("="*60)
    model, tokenizer = load_model_lora()
    train_sft(model, tokenizer, tag="sft_phase")

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_grpo_curriculum(model, tokenizer, tag="sft_grpo", mix_previous=True)
    FastLanguageModel.for_inference(model)

    print("\nEvaluating SFT->GRPO...")
    sft_grpo_results = {}
    for lvl in range(1, MAX_LEVEL + 1):
        r = eval_pass_at_k(model, tokenizer, lvl, tag="sft_grpo")
        sft_grpo_results[lvl] = r
        print(f"  L{lvl} pass@1={r['pass_at_k'].get(1,0):.4f} "
              f"pass@128={r['pass_at_k'].get(128,0):.4f}")
    all_results["sft_grpo"] = sft_grpo_results

    with open(LOGS_DIR / "sft_grpo_pass_at_k.json", "w") as f:
        json.dump({str(k): v["pass_at_k"] for k,v in sft_grpo_results.items()}, f, indent=2)

    del model, tokenizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Step 3: SRFT ---
    print("\n" + "="*60)
    print("STEP 3: SRFT")
    print("="*60)
    model, tokenizer = load_model_lora()
    train_srft(model, tokenizer, tag="srft", mix_previous=True)
    FastLanguageModel.for_inference(model)

    print("\nEvaluating SRFT...")
    srft_results = {}
    for lvl in range(1, MAX_LEVEL + 1):
        r = eval_pass_at_k(model, tokenizer, lvl, tag="srft")
        srft_results[lvl] = r
        print(f"  L{lvl} pass@1={r['pass_at_k'].get(1,0):.4f} "
              f"pass@128={r['pass_at_k'].get(128,0):.4f}")
    all_results["srft"] = srft_results

    with open(LOGS_DIR / "srft_pass_at_k.json", "w") as f:
        json.dump({str(k): v["pass_at_k"] for k,v in srft_results.items()}, f, indent=2)

    # --- Save all results ---
    with open(LOGS_DIR / "all_results.json", "w") as f:
        merged = {}
        for method, res in all_results.items():
            merged[method] = {str(k): v["pass_at_k"] for k,v in res.items()}
        json.dump(merged, f, indent=2)

    print("\n  All results saved to", LOGS_DIR)
    return all_results

# ============================================================
# 11) ANALYSIS
# ============================================================
def run_analysis():
    """Графики и таблицы для отчёта."""
    import matplotlib.pyplot as plt

    methods = {}
    for tag in ["grpo_only", "sft_grpo", "srft"]:
        p = LOGS_DIR / f"{tag}_pass_at_k.json"
        if p.exists():
            with open(p) as f:
                methods[tag] = json.load(f)

    if not methods:
        print("Нет результатов. Сначала запустите run_full_experiment().")
        return

    k_values = [1, 4, 8, 16, 32, 64, 128]

    # --- 1. pass@k curves: all levels + hard ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title("pass@k on Val (all levels avg)")
    for name, data in methods.items():
        avg_pk = []
        for k in k_values:
            vals = [data.get(str(l),{}).get(str(k),0) for l in range(1, MAX_LEVEL+1)]
            avg_pk.append(np.mean(vals) if vals else 0)
        ax.plot(k_values, avg_pk, marker="o", label=name)
    ax.set_xlabel("k"); ax.set_ylabel("pass@k"); ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.set_title(f"pass@k on Hard (levels {HARD_LEVELS})")
    for name, data in methods.items():
        avg_pk = []
        for k in k_values:
            vals = [data.get(str(l),{}).get(str(k),0) for l in HARD_LEVELS]
            avg_pk.append(np.mean(vals) if vals else 0)
        ax.plot(k_values, avg_pk, marker="o", label=name)
    ax.set_xlabel("k"); ax.set_ylabel("pass@k"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(LOGS_DIR / "pass_at_k_comparison.png", dpi=150)
    plt.show()

    # --- 2. Per-level table ---
    print("\n" + "="*70)
    print("PASS@128 PER LEVEL")
    print("="*70)
    header = f"{'Level':>6}"
    for name in methods: header += f" | {name:>12}"
    print(header)
    print("-"*70)
    for lvl in range(1, MAX_LEVEL + 1):
        row = f"  L{lvl:2d}  "
        for name, data in methods.items():
            v = data.get(str(lvl),{}).get("128",0)
            row += f" | {v:12.4f}"
        print(row)

    # --- 3. "Сломали ноль?" ---
    print("\n" + "="*60)
    print(f"HARD LEVELS ({HARD_LEVELS}): pass@128 > 0?")
    print("="*60)
    for name, data in methods.items():
        hard_pk = [data.get(str(l),{}).get("128",0) for l in HARD_LEVELS]
        broken = sum(1 for v in hard_pk if v > 0)
        print(f"  {name}: {broken}/{len(HARD_LEVELS)} levels broken zero "
              f"(avg pass@128={np.mean(hard_pk):.4f})")

    # --- 4. Reward curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, tag in enumerate(["grpo_only", "sft_grpo", "srft"]):
        ax = axes[idx]
        ax.set_title(f"Reward: {tag}")
        steps_all, rewards_all = [], []
        offset = 0
        for lvl in range(1, MAX_LEVEL + 1):
            p = LOGS_DIR / f"{tag}_metrics_L{lvl}.jsonl"
            if not p.exists(): continue
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    if "reward" in rec:
                        steps_all.append(rec["step"] + offset)
                        rewards_all.append(rec["reward"])
                if steps_all: offset = steps_all[-1] + 10
        if steps_all:
            ax.plot(steps_all, rewards_all, alpha=0.7)
        ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.grid(True)

    plt.tight_layout()
    plt.savefig(LOGS_DIR / "reward_curves.png", dpi=150)
    plt.show()

    # --- 5. Unsolved task analysis ---
    print("\n" + "="*60)
    print("UNSOLVED TASKS ANALYSIS")
    print("="*60)
    for tag in methods:
        print(f"\n--- {tag} ---")
        for lvl in HARD_LEVELS:
            diag_path = LOGS_DIR / f"{tag}_diag_L{lvl}.json"
            if not diag_path.exists(): continue
            with open(diag_path) as f:
                diag = json.load(f)
            unsolved = [t for t in diag.get("per_task",[]) if not t.get("solved")]
            total = len(diag.get("per_task",[]))
            print(f"  L{lvl}: {len(unsolved)}/{total} unsolved")
            for t in unsolved[:5]:
                print(f"    nums={t.get('numbers')} target={t.get('target')} "
                      f"gold={t.get('gold_expr')}")

    print(f"\n  Graphs saved to {LOGS_DIR}/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Target24 Full Experiment")
    print("="*60)
    print("Usage:")
    print("  1. Run check_hard_v2.ipynb first (data + gold)")
    print("  2. Then: run_full_experiment()")
    print("  3. Then: run_analysis()")
    print()
    print("Or run steps individually:")
    print("  from GRPO_full_experiment import *")
    print("  model, tokenizer = load_model_lora()")
    print("  train_grpo_curriculum(model, tokenizer, tag='grpo_only')")

