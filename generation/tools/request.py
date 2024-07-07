import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


g_input_token_cost_usd_by_model = {
    "gpt-4-1106-preview": 0.01 / 1000,
    "gpt-4": 0.03 / 1000,
    "gpt-4-0314": 0.03 / 1000,
    "gpt-4-0613": 0.03 / 1000,
    "gpt-4-32k": 0.06 / 1000,
    "gpt-3.5-turbo": 0.001 / 1000,
    "gpt-3.5-turbo-0301": 0.001 / 1000,
    "gpt-3.5-turbo-0613": 0.001 / 1000,
    "gpt-3.5-turbo-1106": 0.001 / 1000,
    "gpt-3.5-turbo-instruct": 0.0015 / 1000,
    "gpt-3.5-turbo-16k": 0.003 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

g_output_token_cost_usd_by_model = {
    "gpt-4-1106-preview": 0.03 / 1000,
    "gpt-4": 0.06 / 1000,
    "gpt-4-0314": 0.06 / 1000,
    "gpt-4-0613": 0.06 / 1000,
    "gpt-4-32k": 0.12 / 1000,
    "gpt-3.5-turbo": 0.002 / 1000,
    "gpt-3.5-turbo-0301": 0.002 / 1000,
    "gpt-3.5-turbo-0613": 0.002 / 1000,
    "gpt-3.5-turbo-1106": 0.002 / 1000,
    "gpt-3.5-turbo-instruct": 0.002 / 1000,
    "gpt-3.5-turbo-16k": 0.004 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

def request(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 4000,
    temperature: float = 0,
    model: str = "gpt-4",
) -> str:
    while True:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            break
        except:
            print('Connection Error\nRetrying...')
            time.sleep(20)

    final_string = res.choices[0].message.content
    num_input_tokens = res.usage.prompt_tokens
    num_output_tokens = res.usage.completion_tokens
    price = g_input_token_cost_usd_by_model[model] * num_input_tokens + g_output_token_cost_usd_by_model[model] * num_output_tokens
    return final_string, price



def req(prompt, system_prompt = "", model = "gpt-3.5-turbo-16k", tokens = 1024, temperature=0.0):
    return request(
            prompt,
            system_prompt=system_prompt,
            max_tokens=tokens,
            temperature=temperature,
            model=model,
        )