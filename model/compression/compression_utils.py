"""Helper functions for the compress_granite notebook."""

from pathlib import Path

from transformers import AutoModelForCausalLM


def load_model(path):
    """Load a model from disk."""
    return AutoModelForCausalLM.from_pretrained(path, dtype="auto", low_cpu_mem_usage=True)


def generate(model, tokenizer, prompts, max_new_tokens=40):
    """Send each prompt to the model and return the answers."""
    answers = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return answers


def weights_size_mb(path):
    """Return the size of the model weights files (*.safetensors) in MB."""
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1024**2


def print_answers(prompts, answers):
    """Print each prompt with its answer."""
    for prompt, answer in zip(prompts, answers):
        print(f"Q: {prompt}\nA: {answer}\n")


def reload_and_test(save_dir, tokenizer, prompts):
    """Reload a saved model from disk, verify that it answers the prompts, and return the results."""
    model = load_model(save_dir)
    answers = generate(model, tokenizer, prompts)
    del model

    result = {"path": save_dir, "size_mb": weights_size_mb(save_dir), "answers": answers}
    print(f"✓ Model saved to {save_dir}/ ({result['size_mb']:.0f} MB)\n")
    print_answers(prompts, answers)
    return result
