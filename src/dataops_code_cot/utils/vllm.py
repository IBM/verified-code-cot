import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLMClient:
    def __init__(
        self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", max_model_len=4096
    ):
        print(
            f"Initializing VLLMClient with model_name={model_name}, max_model_len={max_model_len}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # Explicitly set 6 GPUs
        llm_args = {
            "model": model_name,
            "gpu_memory_utilization": 0.7,
            "max_model_len": max_model_len,
            "tensor_parallel_size": 4,
            "disable_custom_all_reduce": True,  # Fallback to NCCL
        }
        print(f"Creating LLM with args: {llm_args}")
        try:
            self.llm = LLM(**llm_args)
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            raise
        self.default_sampling_params = SamplingParams(max_tokens=2000)

    def get_model_response(
        self,
        system_prompt,
        user_prompt,
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=2000,
        min_new_tokens=30,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    ):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        if isinstance(system_prompt, str):
            # Single prompt case
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            res = self.llm.chat(messages=messages, sampling_params=sampling_params)
            return res[0].outputs[0].text
        else:
            # Batch of prompts case
            messages_list = [
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                for sys, usr in zip(system_prompt, user_prompt)
            ]
            texts = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_list
            ]
            outputs = self.llm.generate(texts, sampling_params)
            return [output.outputs[0].text for output in outputs]
