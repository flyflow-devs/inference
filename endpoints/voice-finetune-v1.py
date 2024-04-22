import time
from pathlib import Path
import json
import modal
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4

stub = modal.Stub("voice-v1")
model_volume = modal.Volume.from_name("example-runs-vol2")

N_INFERENCE_GPU = 1

image = modal.Image.debian_slim().pip_install("torch", "transformers", "fastapi", "uvicorn", "sentencepiece", "vllm", "tiktoken")

with image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    import tiktoken

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

def get_model_path(model_name: str) -> Path:
    return Path(f"/model/{model_name}")

def format_messages(messages: List[Message]):
    prompt = ""
    for message in messages:
        if message.role == "user" or message.role == "system":
            prompt += "[INST]" + message.content + "[/INST]"
        else:
            prompt += message.content
    return prompt

@stub.cls(
    gpu=modal.gpu.H100(count=N_INFERENCE_GPU),
    image=image,
    secret=modal.Secret.from_name("huggingface-secret"),
    volumes={"/model": model_volume},
    allow_concurrent_inputs=30,
    container_idle_timeout=900,
)
class ChatCompletions:
    def __init__(self, model_name: str = "axo-2024-04-20-18-16-53-4814/lora-out/merged") -> None:
        self.model_name = model_name

    @modal.enter()
    def init(self):
        model_path = get_model_path(self.model_name)
        print("Initializing vLLM engine on:", model_path)

        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=N_INFERENCE_GPU,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    async def _stream(self, input: ChatRequest):
        if not input.messages:
            return

        # Extract the text input from the ChatRequest object
        text_input = format_messages(input.messages)

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=32000,
        )
        request_id = str(uuid4())
        results_generator = self.engine.generate(text_input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                    request_output.outputs[0].text
                    and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    async def non_streaming(self, input: ChatRequest):
        output = [text async for text in self._stream(input)]
        response_text = "".join(output)

        prompt_tokens = self.num_tokens_from_string(format_messages(input.messages), "cl100k_base")
        completion_tokens = self.num_tokens_from_string(response_text, "cl100k_base")
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=int(time.time()),
            model=self.model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )

    @modal.web_endpoint(method="POST", label="voice")
    async def web(self, input: ChatRequest):
        response = await self.non_streaming(input)
        return JSONResponse(response.dict())

@stub.local_entrypoint()
def main(model_name: str = "axo-2024-04-18-21-09-26-4ff0/lora-out/merged"):
    stub.deploy()