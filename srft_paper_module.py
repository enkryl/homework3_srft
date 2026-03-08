"""
srft_paper_module.py

Отдельный модуль с paper-style SRFT поверх существующего GRPO_full_experiment.

Что добавляет:
  - train_srft_paper(...): single-stage SRFT-подобное обучение
  - eval_pass_at_k(...): расширенная оценка с логированием raw / think / answer
    для каждого сгенерированного сэмпла

Использование в ноутбуке:
    import os, sys
    sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))
    from srft_paper_module import *

    model, tokenizer = load_model_lora()
    train_srft_paper(model, tokenizer, tag='srft_paper', mix_previous=True)

    FastLanguageModel.for_inference(model)
    for lvl in range(1, MAX_LEVEL + 1):
        r = eval_pass_at_k(model, tokenizer, lvl, tag='srft_paper')
        print(lvl, r['pass_at_k'])
"""

from GRPO_full_experiment import *  # noqa: F401,F403
from GRPO_full_experiment import _extract_expr

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import get_cosine_schedule_with_warmup


SRFT_PAPER_CFG = {
    "learning_rate": 3e-6,
    "weight_decay": 0.0,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "demo_batch_size": 2,
    "rollout_batch_size": 1,
    "num_generations": NUM_GENERATIONS,
    "max_prompt_length": 256,
    "max_completion_length": MAX_NEW_TOKENS,
    "clip_eps": 0.2,
    "w_sft_base": 0.5,
    "w_rl_base": 0.1,
    "log_rollouts_every": 10,
    "max_logged_rollouts_per_step": 8,
}


def _append_jsonl(path: Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sample_dataset_items(ds, n: int):
    if len(ds) == 0:
        return []
    return [ds[random.randrange(len(ds))] for _ in range(n)]


def _last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[-1].get("content", "") if messages else ""



def _last_assistant_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""



def _prompt_text_from_messages(tokenizer, messages):
    if messages and messages[-1].get("role") == "assistant":
        prompt_messages = messages[:-1]
    else:
        prompt_messages = messages
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )



def _prompt_text_from_prompt_field(tokenizer, prompt_messages):
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )



def _build_answer_meta_lookup(levels=None):
    if levels is None:
        levels = list(range(1, MAX_LEVEL + 1))
    lookup = {}
    for lvl in levels:
        ds = get_dataset_for_level("train", lvl)
        for ex in ds:
            prompt = ex.get("prompt", [])
            question = prompt[-1]["content"] if prompt else ""
            lookup[question] = {
                "answer": ex.get("answer", ""),
                "metadata": ex.get("metadata", None),
            }
    return lookup



def get_srft_paper_demo_records(tokenizer, levels=None):
    """
    Обогащает sft_gold answer/metadata из train-датасета.
    Это нужно, чтобы demo-примеры можно было использовать и в SFT-, и в RL-части.
    """
    if levels is None:
        levels = list(range(1, MAX_LEVEL + 1))
    lookup = _build_answer_meta_lookup(levels)
    raw = get_sft_dataset()
    demo_records = []
    dropped = 0

    for ex in raw:
        messages = ex.get("messages", [])
        question = _last_user_message(messages)
        completion = _last_assistant_message(messages)
        meta = lookup.get(question, {})
        answer = ex.get("answer", meta.get("answer", ""))
        metadata = ex.get("metadata", meta.get("metadata", None))

        if not question or not completion or metadata is None or answer in (None, ""):
            dropped += 1
            continue

        demo_records.append({
            "question": question,
            "prompt_text": _prompt_text_from_messages(tokenizer, messages),
            "completion_text": completion,
            "answer": answer,
            "metadata": metadata,
        })

    if not demo_records:
        raise ValueError("SRFT-paper: no usable demonstration records after enrichment from train data.")

    print(f"SRFT-paper: demo records={len(demo_records)}, dropped={dropped}")
    return demo_records



def _scalar_total_reward(prompt_messages, completion_text, answer, metadata):
    comps = [[{"role": "assistant", "content": completion_text}]]
    kwargs = {
        "prompts": [prompt_messages],
        "completions": comps,
        "answer": [answer],
        "metadata": [metadata],
    }
    total = 0.0
    for fn in reward_funcs:
        total += float(fn(**kwargs)[0])
    exact = float(correctness_reward_func(**kwargs)[0])
    return total, exact



def _sequence_stats(model, tokenizer, prompt_text: str, completion_text: str, max_length=None):
    if max_length is None:
        max_length = MAX_SEQ_LEN
    completion_text = completion_text or ""

    tok_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    tok_full = tokenizer(
        prompt_text + completion_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = tok_full["input_ids"].to(model.device)
    attention_mask = tok_full["attention_mask"].to(model.device)
    prompt_len = tok_prompt["input_ids"].shape[1]

    if input_ids.shape[1] <= prompt_len + 1:
        zero = torch.tensor(0.0, device=model.device)
        return zero, zero

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    pos = torch.arange(labels.shape[1], device=model.device).unsqueeze(0)
    completion_mask = pos >= max(prompt_len - 1, 0)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs[completion_mask]

    probs = torch.softmax(logits, dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)
    masked_entropy = token_entropy[completion_mask]

    if masked_log_probs.numel() == 0:
        zero = torch.tensor(0.0, device=model.device)
        return zero, zero

    return masked_log_probs.mean(), masked_entropy.mean()


@torch.inference_mode()
def _generate_rollout_texts(model, tokenizer, prompt_text: str, num_generations=None, max_new_tokens=None):
    if num_generations is None:
        num_generations = SRFT_PAPER_CFG["num_generations"]
    if max_new_tokens is None:
        max_new_tokens = SRFT_PAPER_CFG["max_completion_length"]

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        num_return_sequences=num_generations,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    texts = []
    for seq in out:
        texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    return texts[:num_generations]



def _mean_tensor(values, device):
    if not values:
        return torch.tensor(0.0, device=device)
    return torch.stack(values).mean()



def extract_think_text(text: str) -> str:
    m = THINK_BLOCK.search(text or "")
    return (m.group(0) if m else "").strip().removeprefix("<think>").removesuffix("</think>").strip()



def extract_answer_text(text: str) -> str:
    m = ANSWER_BLOCK.search(text or "")
    return (m.group(1) if m else "").strip()



def _token_len(tokenizer, text: str) -> int:
    text = text or ""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])



def extract_trace_fields(tokenizer, text: str) -> dict:
    think_text = extract_think_text(text)
    answer_text = extract_answer_text(text)
    return {
        "raw_text": text or "",
        "think_text": think_text,
        "answer_text": answer_text,
        "answer_expr": _extract_expr(text or ""),
        "has_think": bool(think_text),
        "has_answer": bool(answer_text),
        "think_n_tokens": _token_len(tokenizer, think_text),
        "answer_n_tokens": _token_len(tokenizer, answer_text),
        "raw_n_tokens": _token_len(tokenizer, text or ""),
    }



def train_srft_paper(model, tokenizer, tag="srft_paper", levels=None, mix_previous=True,
                     cfg=None, steps_override=None):
    """
    Paper-style SRFT: single-stage optimization with
      1) demo SFT loss,
      2) demo RL loss,
      3) self-rollout RL loss,
      4) entropy-aware weights wSFT и wRL.

    В логи дополнительно пишет rollout traces с полями raw_text / think_text / answer_text.
    """
    if levels is None:
        levels = list(range(1, MAX_LEVEL + 1))
    cfg = dict(SRFT_PAPER_CFG if cfg is None else {**SRFT_PAPER_CFG, **cfg})

    demo_records = get_srft_paper_demo_records(tokenizer, levels)
    srft_steps = {k: max(80, STEPS_PER_LEVEL[k] // 2) for k in levels}
    if steps_override:
        srft_steps.update(steps_override)

    total_steps = int(sum(srft_steps[k] for k in levels))

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * cfg["warmup_ratio"])),
        num_training_steps=max(1, total_steps),
    )

    global_step = 0
    all_metrics = []

    for lvl in levels:
        stage_dir = RUNS_DIR / f"{tag}_L{lvl}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{tag}_L{lvl}.log"
        jsonl_log = LOGS_DIR / f"{tag}_metrics_L{lvl}.jsonl"
        traces_log = LOGS_DIR / f"{tag}_traces_L{lvl}.jsonl"
        logger = make_logger(f"{tag}_L{lvl}", log_file)
        logger.info(f"=== SRFT-PAPER LEVEL {lvl} ===")

        if mix_previous and lvl > min(levels):
            dss = [get_dataset_for_level("train", k) for k in levels if k <= lvl]
            rollout_ds = concatenate_datasets(dss).shuffle(seed=SEED)
            logger.info(f"Rollout dataset: mixed levels {min(levels)}..{lvl}, n={len(rollout_ds)}")
        else:
            rollout_ds = get_dataset_for_level("train", lvl)
            logger.info(f"Rollout dataset: level {lvl}, n={len(rollout_ds)}")
        logger.info(f"Demo dataset: n={len(demo_records)}")

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for local_step in range(1, srft_steps[lvl] + 1):
            model.train()

            demo_batch = random.sample(demo_records, k=min(cfg["demo_batch_size"], len(demo_records)))
            rollout_batch = _sample_dataset_items(rollout_ds, cfg["rollout_batch_size"])

            rollout_records = []
            for ex in rollout_batch:
                prompt_messages = ex["prompt"]
                prompt_text = _prompt_text_from_prompt_field(tokenizer, prompt_messages)
                question = prompt_messages[-1]["content"] if prompt_messages else ""

                model.eval()
                completions = _generate_rollout_texts(
                    model,
                    tokenizer,
                    prompt_text,
                    num_generations=cfg["num_generations"],
                    max_new_tokens=cfg["max_completion_length"],
                )
                model.train()

                for comp_idx, comp in enumerate(completions):
                    total_reward, exact_reward = _scalar_total_reward(
                        prompt_messages, comp, ex.get("answer", ""), ex.get("metadata", None)
                    )
                    rollout_records.append({
                        "level": int(lvl),
                        "question": question,
                        "prompt_text": prompt_text,
                        "completion_index": int(comp_idx),
                        "completion_text": comp,
                        "answer": ex.get("answer", ""),
                        "metadata": ex.get("metadata", None),
                        "reward": total_reward,
                        "exact_reward": exact_reward,
                        "is_positive": exact_reward > 0.0,
                    })

            demo_step_records = []
            for ex in demo_batch:
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["question"]},
                ]
                total_reward, exact_reward = _scalar_total_reward(
                    prompt_messages,
                    ex["completion_text"],
                    ex["answer"],
                    ex["metadata"],
                )
                demo_step_records.append({
                    **ex,
                    "reward": total_reward,
                    "exact_reward": exact_reward,
                })

            aug_rewards = [r["reward"] for r in rollout_records] + [r["reward"] for r in demo_step_records]
            reward_mean = float(np.mean(aug_rewards)) if aug_rewards else 0.0
            reward_std = float(np.std(aug_rewards) + 1e-6) if aug_rewards else 1.0

            demo_sft_terms, demo_rl_terms = [], []
            demo_entropies, rollout_entropies = [], []
            pos_terms, neg_terms = [], []

            for ex in demo_step_records:
                mean_logprob, mean_entropy = _sequence_stats(
                    model, tokenizer, ex["prompt_text"], ex["completion_text"]
                )
                demo_entropies.append(mean_entropy)
                adv = (ex["reward"] - reward_mean) / reward_std
                ratio = torch.exp(mean_logprob).clamp(min=1e-6)
                adv_t = torch.tensor(float(adv), device=model.device)
                clipped = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"])
                demo_sft_terms.append(-mean_logprob)
                demo_rl_terms.append(-torch.minimum(ratio * adv_t, clipped * adv_t))

            for ex in rollout_records:
                mean_logprob, mean_entropy = _sequence_stats(
                    model, tokenizer, ex["prompt_text"], ex["completion_text"]
                )
                rollout_entropies.append(mean_entropy)
                if ex["is_positive"]:
                    pos_terms.append(-mean_logprob)
                else:
                    neg_terms.append(mean_logprob)

            demo_entropy = _mean_tensor(demo_entropies, model.device)
            rollout_entropy = _mean_tensor(rollout_entropies, model.device)
            w_sft = cfg["w_sft_base"] * torch.exp(-demo_entropy.detach())
            w_rl = cfg["w_rl_base"] * torch.exp(rollout_entropy.detach())

            demo_sft_loss = w_sft * _mean_tensor(demo_sft_terms, model.device)
            demo_rl_loss = _mean_tensor(demo_rl_terms, model.device)
            rollout_rl_loss = (
                w_rl * _mean_tensor(pos_terms, model.device)
                + _mean_tensor(neg_terms, model.device)
            )
            total_loss = demo_sft_loss + demo_rl_loss + rollout_rl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if local_step % 10 == 0 or local_step == 1 or local_step == srft_steps[lvl]:
                rec = {
                    "step": int(global_step),
                    "level": int(lvl),
                    "local_step": int(local_step),
                    "loss": float(total_loss.detach().cpu()),
                    "demo_sft_loss": float(demo_sft_loss.detach().cpu()),
                    "demo_rl_loss": float(demo_rl_loss.detach().cpu()),
                    "rollout_rl_loss": float(rollout_rl_loss.detach().cpu()),
                    "w_sft": float(w_sft.detach().cpu()),
                    "w_rl": float(w_rl.detach().cpu()),
                    "demo_entropy": float(demo_entropy.detach().cpu()),
                    "rollout_entropy": float(rollout_entropy.detach().cpu()),
                    "rollout_reward_mean": float(np.mean([r["reward"] for r in rollout_records])) if rollout_records else 0.0,
                    "demo_reward_mean": float(np.mean([r["reward"] for r in demo_step_records])) if demo_step_records else 0.0,
                    "n_positive": int(sum(1 for r in rollout_records if r["is_positive"])),
                    "n_negative": int(sum(1 for r in rollout_records if not r["is_positive"])),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                logger.info(
                    f"step={global_step} loss={rec['loss']:.4f} "
                    f"demo_sft={rec['demo_sft_loss']:.4f} demo_rl={rec['demo_rl_loss']:.4f} "
                    f"rollout_rl={rec['rollout_rl_loss']:.4f} wSFT={rec['w_sft']:.4f} "
                    f"wRL={rec['w_rl']:.4f} pos={rec['n_positive']} neg={rec['n_negative']}"
                )
                _append_jsonl(jsonl_log, rec)

            if local_step % cfg["log_rollouts_every"] == 0 or local_step == 1:
                for rr in rollout_records[: cfg["max_logged_rollouts_per_step"]]:
                    trace = extract_trace_fields(tokenizer, rr["completion_text"])
                    _append_jsonl(traces_log, {
                        "step": int(global_step),
                        "level": int(lvl),
                        "local_step": int(local_step),
                        "question": rr["question"],
                        "reward": float(rr["reward"]),
                        "exact_reward": float(rr["exact_reward"]),
                        "is_positive": bool(rr["is_positive"]),
                        **trace,
                    })

            if local_step % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        adapter_dir = stage_dir / "lora_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"Train done in {dt/60:.1f} min")
        logger.info(f"Saved adapter: {adapter_dir}")
        logger.info(f"Saved rollout traces: {traces_log}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_metrics.append({
            "level": lvl,
            "train_time_min": dt / 60,
            "steps": srft_steps[lvl],
        })

    final = RUNS_DIR / f"{tag}_final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"  {tag} final LoRA saved: {final}")
    return all_metrics



def eval_pass_at_k(model, tokenizer, level, limit=EVAL_LIMIT,
                   k_values=None, tag="model", use_vllm=None,
                   save_diagnostics=True, save_samples_jsonl=True,
                   max_logged_samples=None):
    """
    Drop-in replacement для base eval_pass_at_k, но теперь сохраняет полные traces:
      - raw_text
      - think_text
      - answer_text
      - answer_expr
      - token counts
      - correctness per sample

    Файлы:
      - {tag}_diag_L{level}.json      : агрегированная диагностика + traces по задачам
      - {tag}_samples_L{level}.jsonl  : один completion на строку, удобно фильтровать
    """
    if k_values is None:
        k_values = [1, 4, 8, 16, 32, 64, 128]
    if use_vllm is None:
        use_vllm = USE_VLLM
    if max_logged_samples is None:
        max_logged_samples = N_SAMPLES_PASSK

    val_path = DATA_DIR / f"val_L{level}.jsonl"
    items = Data.from_jsonl_file(str(val_path))
    subset = items[:limit] if limit else items

    gen_fn = generate_samples_vllm if use_vllm else generate_samples_hf
    per_task = []
    samples_jsonl_path = LOGS_DIR / f"{tag}_samples_L{level}.jsonl"
    if save_samples_jsonl:
        samples_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        open(samples_jsonl_path, "w", encoding="utf-8").close()

    for i, d in enumerate(subset):
        comps = gen_fn(model, tokenizer, d.question, N_SAMPLES_PASSK)
        logged_traces = []
        n_correct = 0

        for sample_idx, c in enumerate(comps):
            is_correct = bool(env.verify(d, c))
            n_correct += int(is_correct)
            trace = {
                "task_idx": int(i),
                "sample_idx": int(sample_idx),
                "level": int(level),
                "question": d.question,
                "numbers": d.metadata.get("numbers"),
                "target": d.metadata.get("target"),
                "gold_expr": d.metadata.get("gold_expr"),
                "is_correct": is_correct,
                **extract_trace_fields(tokenizer, c),
            }
            if sample_idx < max_logged_samples:
                logged_traces.append(trace)
            if save_samples_jsonl:
                _append_jsonl(samples_jsonl_path, trace)

        per_task.append({
            "idx": int(i),
            "n_correct": int(n_correct),
            "n_total": int(N_SAMPLES_PASSK),
            "numbers": d.metadata.get("numbers"),
            "target": d.metadata.get("target"),
            "gold_expr": d.metadata.get("gold_expr"),
            "solved": n_correct > 0,
            "samples": logged_traces,
        })

        if (i + 1) % 10 == 0:
            solved_so_far = sum(1 for t in per_task if t["solved"])
            print(
                f"  {tag} L{level} [{i+1}/{len(subset)}] "
                f"this={n_correct}/{N_SAMPLES_PASSK} "
                f"solved={solved_so_far}/{len(per_task)}"
            )

    pk = {}
    for k in k_values:
        if k > N_SAMPLES_PASSK:
            continue
        pk[k] = float(np.mean([pass_at_k(t["n_total"], t["n_correct"], k) for t in per_task]))

    result = {
        "level": int(level),
        "pass_at_k": pk,
        "per_task": per_task,
        "tag": tag,
        "samples_jsonl": str(samples_jsonl_path) if save_samples_jsonl else None,
    }

    solved = [t for t in per_task if t["solved"]]
    unsolved = [t for t in per_task if not t["solved"]]

    print(f"\n  L{level} SUMMARY: {len(solved)}/{len(per_task)} solved (pass@128={pk.get(128,0):.4f})")
    if unsolved:
        print(f"  Unsolved tasks ({len(unsolved)}):")
        for t in unsolved[:10]:
            print(f"    #{t['idx']}: numbers={t['numbers']} target={t['target']} gold={t['gold_expr']}")
        if len(unsolved) > 10:
            print(f"    ... и ещё {len(unsolved)-10}")

    if save_diagnostics:
        diag_path = LOGS_DIR / f"{tag}_diag_L{level}.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved diagnostics: {diag_path}")
        if save_samples_jsonl:
            print(f"  Saved sample traces: {samples_jsonl_path}")

    return result
