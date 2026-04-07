max_token_val_client = {}
max_token_val_client["rits"] = {}
max_token_val_client["vllm"] = {}

prompt_types = [
    "summary",
    "io",
    "question",
    "forward",
    "backward",
    "forward_feedback",
    "backward_feedback",
]

default_max_tokens = 2000

# Set max_new_tokens value for VLLM Client
for prompt in prompt_types:
    max_token_val_client["vllm"][prompt] = default_max_tokens

# set max_new_tokens values for RITS client
max_token_val_client["rits"] = {
    "summary": 200,
    "io": 200,
    "question": 200,
    "forward": default_max_tokens,
    "backward": default_max_tokens,
    "forward_feedback": 300,
    "backward_feedback": 300,
}
