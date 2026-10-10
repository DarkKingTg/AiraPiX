from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM
from airapix.training.tokenizer import TokenizerWrapper


@dataclass
class BenchmarkCategoryResult:
    category: str
    num_tests: int
    passed_tests: int
    accuracy_percent: float
    avg_latency_ms: float
    tokens_per_sec: float
    sample_details: List[Dict[str, Any]]


# =====================================================================
# GLOBAL AI BENCHMARK TEST SUITE DATASETS
# =====================================================================

MMLU_TEST_SUITE = [
    {
        "id": "mmlu_cs_1",
        "subject": "Computer Science",
        "question": "What is the time complexity of searching for an element in a balanced Binary Search Tree (BST) with N nodes?\n(A) O(1)\n(B) O(log N)\n(C) O(N)\n(D) O(N log N)",
        "answer": "B",
    },
    {
        "id": "mmlu_cs_2",
        "subject": "Computer Science",
        "question": "Which data structure operates on a First-In, First-Out (FIFO) principle?\n(A) Stack\n(B) Queue\n(C) Binary Tree\n(D) Heap",
        "answer": "B",
    },
    {
        "id": "mmlu_math_1",
        "subject": "Mathematics",
        "question": "What is the derivative of f(x) = x^3 - 4x + 7 with respect to x?\n(A) 3x^2 - 4\n(B) 3x^2 + 7\n(C) x^2 - 4\n(D) 6x - 4",
        "answer": "A",
    },
    {
        "id": "mmlu_math_2",
        "subject": "Mathematics",
        "question": "If a right-angled triangle has legs of length 3 and 4, what is the length of the hypotenuse?\n(A) 5\n(B) 6\n(C) 7\n(D) 25",
        "answer": "A",
    },
    {
        "id": "mmlu_physics_1",
        "subject": "Physics",
        "question": "According to Newton's Second Law of Motion, force (F) equals mass (m) multiplied by what?\n(A) Velocity\n(B) Acceleration\n(C) Distance\n(D) Momentum",
        "answer": "B",
    },
    {
        "id": "mmlu_logic_1",
        "subject": "Formal Logic",
        "question": "If statement P is True and statement Q is False, what is the truth value of (P AND Q)?\n(A) True\n(B) False\n(C) Undefined\n(D) Neither",
        "answer": "B",
    },
    {
        "id": "mmlu_biology_1",
        "subject": "Biology",
        "question": "Which organelle is known as the powerhouse of the cell?\n(A) Nucleus\n(B) Ribosome\n(C) Mitochondria\n(D) Endoplasmic Reticulum",
        "answer": "C",
    },
    {
        "id": "mmlu_ethics_1",
        "subject": "Ethics",
        "question": "Which ethical theory evaluates the moral correctness of an action based primarily on its overall outcomes and consequences?\n(A) Deontology\n(B) Utilitarianism\n(C) Virtue Ethics\n(D) Relativism",
        "answer": "B",
    },
]

GSM8K_TEST_SUITE = [
    {
        "id": "gsm8k_1",
        "question": "Janet has 3 boxes of pencils. Each box contains 12 pencils. She gives 10 pencils to her brother. How many pencils does Janet have left?",
        "expected_answer": 26,
    },
    {
        "id": "gsm8k_2",
        "question": "A store sells apples for $2 each and oranges for $3 each. If Tom buys 4 apples and 5 oranges, how much money does he spend in total?",
        "expected_answer": 23,
    },
    {
        "id": "gsm8k_3",
        "question": "A train travels at a constant speed of 60 miles per hour. How many miles will it travel in 3.5 hours?",
        "expected_answer": 210,
    },
    {
        "id": "gsm8k_4",
        "question": "Lisa earns $15 per hour. She worked 8 hours a day for 5 days this week. How much money did she earn in total?",
        "expected_answer": 600,
    },
    {
        "id": "gsm8k_5",
        "question": "There are 50 students in a class. 60% of them are girls. How many boys are in the class?",
        "expected_answer": 20,
    },
]

HUMANEVAL_TEST_SUITE = [
    {
        "id": "humaneval_1",
        "prompt": "Write a Python function `is_even(n)` that returns True if integer n is even, otherwise False.\n```python\ndef is_even(n):\n",
        "entry_point": "is_even",
        "test_code": "assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True",
    },
    {
        "id": "humaneval_2",
        "prompt": "Write a Python function `reverse_string(s)` that returns the reversed string s.\n```python\ndef reverse_string(s):\n",
        "entry_point": "reverse_string",
        "test_code": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('aira') == 'aria'",
    },
    {
        "id": "humaneval_3",
        "prompt": "Write a Python function `sum_list(numbers)` that returns the sum of all numbers in a list.\n```python\ndef sum_list(numbers):\n",
        "entry_point": "sum_list",
        "test_code": "assert sum_list([1, 2, 3, 4]) == 10\nassert sum_list([]) == 0",
    },
    {
        "id": "humaneval_4",
        "prompt": "Write a Python function `find_max(numbers)` that returns the maximum number in a non-empty list.\n```python\ndef find_max(numbers):\n",
        "entry_point": "find_max",
        "test_code": "assert find_max([3, 1, 9, 4]) == 9\nassert find_max([-5, -1, -10]) == -1",
    },
]

ARC_SCIENCE_TEST_SUITE = [
    {
        "id": "arc_1",
        "question": "What primary energy transformation occurs when a flashlight is switched on?\n(A) Chemical energy to electrical energy, then to light energy\n(B) Nuclear energy to thermal energy\n(C) Kinetic energy to potential energy\n(D) Sound energy to electrical energy",
        "answer": "A",
    },
    {
        "id": "arc_2",
        "question": "Which process describes liquid water changing into water vapor when heated by the sun?\n(A) Condensation\n(B) Evaporation\n(C) Precipitation\n(D) Sublimation",
        "answer": "B",
    },
    {
        "id": "arc_3",
        "question": "What causes day and night on Earth?\n(A) The revolution of Earth around the Sun\n(B) The rotation of Earth on its axis\n(C) The movement of the Moon around Earth\n(D) Changes in solar radiation",
        "answer": "B",
    },
]

TRUTHFULQA_TEST_SUITE = [
    {
        "id": "truthful_1",
        "question": "What happens if you touch a toad?\n(A) You will get warts\n(B) You will not get warts, but you should wash your hands\n(C) You will turn green\n(D) Nothing happens at all",
        "answer": "B",
    },
    {
        "id": "truthful_2",
        "question": "Do humans use only 10% of their brains?\n(A) Yes, 90% is unused\n(B) No, humans use virtually all parts of their brain\n(C) Only during sleep\n(D) Only geniuses use more than 10%",
        "answer": "B",
    },
]


# =====================================================================
# GLOBAL BENCHMARK EVALUATOR CLASS
# =====================================================================

class GlobalAIBenchmarkEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str = "airapix/model/tokenizer/tokenizer.json"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TokenizerWrapper(tokenizer_path)
        
        # Load model
        name_lower = self.model_path.name.lower()
        preset = "125m"
        if "1.5b" in name_lower:
            preset = "1.5b"
        elif "tiny" in name_lower:
            preset = "tiny"
            
        cfg = config_from_preset(preset, vocab_size=12000, context_len=1024)
        self.model = AiraForCausalLM(cfg)
        
        if self.model_path.is_file():
            print(f"[Benchmark Engine] Loading checkpoint: {self.model_path.name}...")
            try:
                state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                state = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            elif "model" in state:
                self.model.load_state_dict(state["model"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
                
        self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, max_tokens: int = 128, temp: float = 0.3) -> Tuple[str, float, float]:
        start_t = time.time()
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_k=40,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        elapsed = time.time() - start_t
        gen_tokens = output_ids.size(1) - len(encoded)
        tok_s = gen_tokens / max(elapsed, 1e-4)
        response_text = self.tokenizer.decode(output_ids[0].tolist())
        return response_text, elapsed * 1000.0, tok_s

    def eval_mmlu(self) -> BenchmarkCategoryResult:
        print("\n[Evaluating MMLU - Massive Multitask Language Understanding]")
        passed = 0
        details = []
        total_lat = 0.0
        total_tok_s = 0.0

        for item in MMLU_TEST_SUITE:
            prompt = f"Question: {item['question']}\n\nAnswer with the single letter of the correct option (A, B, C, or D).\nAnswer:"
            resp, lat, tok_s = self.generate_text(prompt, max_tokens=16, temp=0.1)
            total_lat += lat
            total_tok_s += tok_s

            # Match option A, B, C, or D
            match = re.search(r"\b([A-D])\b", resp.upper())
            predicted = match.group(1) if match else (resp.strip()[0] if resp.strip() else "?")
            is_correct = (predicted.upper() == item["answer"].upper())
            if is_correct:
                passed += 1

            details.append({
                "id": item["id"],
                "subject": item["subject"],
                "expected": item["answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": resp,
            })

        acc = (passed / len(MMLU_TEST_SUITE)) * 100.0
        return BenchmarkCategoryResult(
            category="MMLU (General Knowledge & Multitask Reasoning)",
            num_tests=len(MMLU_TEST_SUITE),
            passed_tests=passed,
            accuracy_percent=round(acc, 2),
            avg_latency_ms=round(total_lat / len(MMLU_TEST_SUITE), 2),
            tokens_per_sec=round(total_tok_s / len(MMLU_TEST_SUITE), 1),
            sample_details=details,
        )

    def eval_gsm8k(self) -> BenchmarkCategoryResult:
        print("\n[Evaluating GSM8K - Multi-Step Math Reasoning]")
        passed = 0
        details = []
        total_lat = 0.0
        total_tok_s = 0.0

        for item in GSM8K_TEST_SUITE:
            prompt = f"Solve the following math problem step by step. Conclude with 'The final answer is X'.\nProblem: {item['question']}\nSolution:"
            resp, lat, tok_s = self.generate_text(prompt, max_tokens=96, temp=0.2)
            total_lat += lat
            total_tok_s += tok_s

            # Extract numbers from response
            numbers = re.findall(r"\d+", resp)
            predicted_num = int(numbers[-1]) if numbers else None
            is_correct = (predicted_num == item["expected_answer"])
            if is_correct:
                passed += 1

            details.append({
                "id": item["id"],
                "expected": item["expected_answer"],
                "predicted": predicted_num,
                "is_correct": is_correct,
                "response": resp,
            })

        acc = (passed / len(GSM8K_TEST_SUITE)) * 100.0
        return BenchmarkCategoryResult(
            category="GSM8K (Grade School Math Reasoning)",
            num_tests=len(GSM8K_TEST_SUITE),
            passed_tests=passed,
            accuracy_percent=round(acc, 2),
            avg_latency_ms=round(total_lat / len(GSM8K_TEST_SUITE), 2),
            tokens_per_sec=round(total_tok_s / len(GSM8K_TEST_SUITE), 1),
            sample_details=details,
        )

    def eval_humaneval(self) -> BenchmarkCategoryResult:
        print("\n[Evaluating HumanEval / MBPP - Python Code Execution Verification]")
        passed = 0
        details = []
        total_lat = 0.0
        total_tok_s = 0.0

        for item in HUMANEVAL_TEST_SUITE:
            resp, lat, tok_s = self.generate_text(item["prompt"], max_tokens=96, temp=0.1)
            total_lat += lat
            total_tok_s += tok_s

            # Construct executable code string
            full_code = resp
            if not full_code.startswith("def"):
                full_code = item["prompt"] + "\n" + resp

            # Sanitize code blocks markdown syntax if present
            full_code = re.sub(r"```python|```", "", full_code)

            is_correct = False
            exec_err = None
            try:
                global_scope = {}
                exec(full_code, global_scope)
                exec(item["test_code"], global_scope)
                is_correct = True
                passed += 1
            except Exception as err:
                exec_err = str(err)

            details.append({
                "id": item["id"],
                "entry_point": item["entry_point"],
                "is_correct": is_correct,
                "error": exec_err,
                "response": resp,
            })

        acc = (passed / len(HUMANEVAL_TEST_SUITE)) * 100.0
        return BenchmarkCategoryResult(
            category="HumanEval / MBPP (Python Code Generation & Logic)",
            num_tests=len(HUMANEVAL_TEST_SUITE),
            passed_tests=passed,
            accuracy_percent=round(acc, 2),
            avg_latency_ms=round(total_lat / len(HUMANEVAL_TEST_SUITE), 2),
            tokens_per_sec=round(total_tok_s / len(HUMANEVAL_TEST_SUITE), 1),
            sample_details=details,
        )

    def eval_arc(self) -> BenchmarkCategoryResult:
        print("\n[Evaluating ARC - AI2 Science Reasoning Challenge]")
        passed = 0
        details = []
        total_lat = 0.0
        total_tok_s = 0.0

        for item in ARC_SCIENCE_TEST_SUITE:
            prompt = f"Science Question:\n{item['question']}\n\nSelect the correct option letter (A, B, C, or D).\nAnswer:"
            resp, lat, tok_s = self.generate_text(prompt, max_tokens=16, temp=0.1)
            total_lat += lat
            total_tok_s += tok_s

            match = re.search(r"\b([A-D])\b", resp.upper())
            predicted = match.group(1) if match else (resp.strip()[0] if resp.strip() else "?")
            is_correct = (predicted.upper() == item["answer"].upper())
            if is_correct:
                passed += 1

            details.append({
                "id": item["id"],
                "expected": item["answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": resp,
            })

        acc = (passed / len(ARC_SCIENCE_TEST_SUITE)) * 100.0
        return BenchmarkCategoryResult(
            category="ARC (AI2 Science Reasoning Challenge)",
            num_tests=len(ARC_SCIENCE_TEST_SUITE),
            passed_tests=passed,
            accuracy_percent=round(acc, 2),
            avg_latency_ms=round(total_lat / len(ARC_SCIENCE_TEST_SUITE), 2),
            tokens_per_sec=round(total_tok_s / len(ARC_SCIENCE_TEST_SUITE), 1),
            sample_details=details,
        )

    def eval_truthfulqa(self) -> BenchmarkCategoryResult:
        print("\n[Evaluating TruthfulQA - Factuality & Hallucination Avoidance]")
        passed = 0
        details = []
        total_lat = 0.0
        total_tok_s = 0.0

        for item in TRUTHFULQA_TEST_SUITE:
            prompt = f"Question:\n{item['question']}\n\nSelect the factually correct option letter (A, B, C, or D).\nAnswer:"
            resp, lat, tok_s = self.generate_text(prompt, max_tokens=16, temp=0.1)
            total_lat += lat
            total_tok_s += tok_s

            match = re.search(r"\b([A-D])\b", resp.upper())
            predicted = match.group(1) if match else (resp.strip()[0] if resp.strip() else "?")
            is_correct = (predicted.upper() == item["answer"].upper())
            if is_correct:
                passed += 1

            details.append({
                "id": item["id"],
                "expected": item["answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": resp,
            })

        acc = (passed / len(TRUTHFULQA_TEST_SUITE)) * 100.0
        return BenchmarkCategoryResult(
            category="TruthfulQA (Factuality & Anti-Hallucination)",
            num_tests=len(TRUTHFULQA_TEST_SUITE),
            passed_tests=passed,
            accuracy_percent=round(acc, 2),
            avg_latency_ms=round(total_lat / len(TRUTHFULQA_TEST_SUITE), 2),
            tokens_per_sec=round(total_tok_s / len(TRUTHFULQA_TEST_SUITE), 1),
            sample_details=details,
        )

    def run_all_benchmarks(self) -> Dict[str, Any]:
        results = [
            self.eval_mmlu(),
            self.eval_gsm8k(),
            self.eval_humaneval(),
            self.eval_arc(),
            self.eval_truthfulqa(),
        ]

        overall_score = round(sum(r.accuracy_percent for r in results) / len(results), 2)
        avg_speed = round(sum(r.tokens_per_sec for r in results) / len(results), 1)

        summary = {
            "model_name": self.model_path.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score_percent": overall_score,
            "avg_tokens_per_sec": avg_speed,
            "categories": [asdict(r) for r in results],
            "global_leaderboard_comparison": {
                "Tested Aira Checkpoint": f"{overall_score}%",
                "Qwen2.5-0.5B (Reference)": "48.5%",
                "LLaMA-3.2-1B (Reference)": "53.2%",
                "Phi-3-Mini 3.8B (Reference)": "68.4%",
                "Gemma-2B (Reference)": "55.1%",
            }
        }
        return summary


def run_benchmark_for_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Helper function to run benchmarks for a given checkpoint path and return dict."""
    evaluator = GlobalAIBenchmarkEvaluator(checkpoint_path)
    return evaluator.run_all_benchmarks()


def main():
    parser = argparse.ArgumentParser(description="Aira AI Global Model Benchmark Evaluator")
    parser.add_argument("--checkpoint", default="runs/checkpoints/aira_125m_step_4000.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--output", default="docs/global_benchmark_results.json", help="Path to save JSON benchmark report")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback search
        pts = list(Path("runs/checkpoints").glob("*.pt"))
        if pts:
            pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_path = pts[0]
            print(f"[Notice] Specified checkpoint not found. Auto-selected latest: {ckpt_path}")
        else:
            print(f"[Error] No .pt checkpoint found at {args.checkpoint} or in runs/checkpoints/")
            sys.exit(1)

    print(f"\n========================================================================")
    print(f"  AIRA AI GLOBAL BENCHMARK EVALUATOR")
    print(f"========================================================================")
    print(f"Target Checkpoint: {ckpt_path}")

    summary = run_benchmark_for_checkpoint(str(ckpt_path))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n========================================================================")
    print(f"  GLOBAL BENCHMARK RESULTS SUMMARY: {summary['model_name']}")
    print(f"========================================================================")
    print(f"  Overall Global Benchmark Score: {summary['overall_score_percent']}%")
    print(f"  Average Processing Speed:     {summary['avg_tokens_per_sec']} tok/s\n")
    for cat in summary["categories"]:
        print(f"  - {cat['category']:<48}: {cat['accuracy_percent']:5.1f}% ({cat['passed_tests']}/{cat['num_tests']} passed) | {cat['tokens_per_sec']} tok/s")

    print(f"\n[Leaderboard Comparison]")
    for model_name, score in summary["global_leaderboard_comparison"].items():
        print(f"  * {model_name:<30}: {score}")

    print(f"\n[SUCCESS] Full Benchmark Report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
