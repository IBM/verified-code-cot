from openai import OpenAI
from transformers import AutoTokenizer

PHI4 = "microsoft-phi-4"
RITS_API_VERSION = "v1"
TOKENIZER = "microsoft/phi-4"


class OpenAIClient:
    def __init__(
        self,
        base_url,
        api_key,
        default_headers=None,
        model_id=None,
    ):
        self.model_id = model_id
        self.llm = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def get_model_response(
        self,
        system_prompt,
        user_prompt,
        model_id=None,
        max_new_tokens=2000,
        temperature=0.5,
        top_k=40,
        top_p=0.9,
    ):
        if model_id is None:
            model_id = self.model_id
        print(model_id)
        # Setup the sampling parameters for generation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if isinstance(system_prompt, str) and isinstance(user_prompt, str):
            response = self.llm.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content.strip()

        elif isinstance(system_prompt, list) and isinstance(user_prompt, list):
            if len(system_prompt) != len(user_prompt):
                raise ValueError(
                    "system_prompt and user_prompt lists must have the same length"
                )

            messages = [
                {"role": "system", "content": sys_msg} for sys_msg in system_prompt
            ] + [{"role": "user", "content": usr_msg} for usr_msg in user_prompt]
            responses = self.llm.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results = [
                responses.choices[i].message.content.strip()
                for i in range(len(responses.choices))
            ]
            return results

        else:
            raise ValueError(
                "system_prompt and user_prompt must both be strings or both be lists"
            )
