"""Evaluate distilled models on AIME, MBPP, LiveCodeBench."""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_aime(model, tokenizer, data_path: str, greedy: bool = True):
    """Evaluate AIME pass@1 (greedy decoding)."""
    with open(data_path) as f:
        problems = [json.loads(line) for line in f]

    correct = 0
    for prob in problems:
        prompt = prob["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, do_sample=not greedy,
                temperature=0.0 if greedy else 0.7,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract answer (problem-specific parsing)
        predicted = extract_answer(response)
        if predicted == prob["answer"]:
            correct += 1

    return correct / len(problems) * 100


def extract_answer(response: str) -> str:
    """Extract numerical answer from model response."""
    # Look for boxed answer or final number
    import re
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"\b(\d+)\b", response)
    return numbers[-1] if numbers else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, choices=["aime", "mbpp", "livecodebench"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    if args.benchmark == "aime":
        score = evaluate_aime(model, tokenizer, args.data)
        print(f"AIME pass@1: {score:.1f}%")
        result = {"benchmark": "aime", "pass@1": score}
    else:
        raise NotImplementedError(f"Benchmark {args.benchmark} not yet implemented")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
