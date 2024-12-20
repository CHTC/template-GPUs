import json
from pathlib import Path
from typing import Generator

from vllm import LLM, SamplingParams


def load_data(
    file: Path, batch_size: int
) -> Generator[list[dict[str, str]], None, None]:
    """Load data from a jsonl file as a generator. Assuming at least `id` and `input` fields are present.

    Example input format: (docs: https://jsonlines.org/)
    {"id": "q0001", "input": "What is the capital of France?"}
    {"id": "q0002", "input": "What is the capital of Germany?"}
    """

    with open(file, mode="r") as f:
        batch = []
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch.clear()
        if batch:
            yield batch


def format_prompt(user_message: str, system_message: str | None = None) -> str:
    """This function formats the user message into the correct format for the `Phi-3.5-mini-instruct` model.

    docs: https://huggingface.co/microsoft/Phi-3.5-mini-instruct#input-formats
    """

    if system_message is None:
        system_message = (
            "You are a helpful assistant. Provide clear and short responses."
        )
    return (
        "<|system|>\n"
        f"{system_message}<|end|>\n"
        "<|user|>\n"
        f"{user_message}<|end|>\n"
        "<|assistant|>\n"
    )


def inference(
    llm: LLM,
    data: list[dict[str, str]],
    input_field: str = "input",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> list[dict]:
    """Perform a batch of inference on Phi-3.5-mini-instruct via vllm offline mode.

    docs: https://docs.vllm.ai/en/v0.6.4/getting_started/examples/offline_inference.html
    """

    inputs = [item[input_field] for item in data]
    formatted_inputs = [format_prompt(user_message=x) for x in inputs]

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts=formatted_inputs, sampling_params=sampling_params)
    text_outputs = [raw_output.outputs[0].text.strip() for raw_output in outputs]
    return [{**item, "output": output} for item, output in zip(data, text_outputs)]


def save_data(data: list[dict], file: Path) -> None:
    """Save data to a jsonl file in append mode.

    Example output format:
    {"id": "q0001", "input": "What is the capital of France?", "output": "Paris."}
    {"id": "q0002", "input": "What is the capital of Germany?", "output": "Berlin."}
    """
    with open(file, mode="a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# Main script
# Perform mini-batch inference on the input data and save the results.
# Adjust the batch size based on your requirements.
# Note: Larger batch sizes may require more GPU memory but can be faster.

data = load_data(Path("inputs.jsonl"), batch_size=20)
llm = LLM(model="microsoft/Phi-3.5-mini-instruct", max_model_len=8192)
for batch in data:
    outputs = inference(llm=llm, data=batch)
    save_data(outputs, Path("outputs.jsonl"))
    print(f"Processed {len(batch)} examples.")
