import sys
from tools.request import req


def simulate(input: str, grounding: str, base_model="gpt-4-1106-preview", temperature=0.0) -> str:
    price = 0
    prompt = """Take a deep breath and work on this problem step by step. 
Given a Dialogue History and Knowledge, your job is to follow instructions in the Knowledge and generate a faithful Response based on the Knowledge and Dialogue History.

#Knowledge#: 
{grounding}

#Dialogue History#: 
{input}

Now, please generate your response:
#Response#: """
    prompt_filled = prompt.replace("{grounding}", grounding).replace("{input}", input)
    response, price_ = req(prompt_filled, tokens=1024, model=base_model, temperature=temperature)
    price += price_
    print("#" * 100)
    print(prompt_filled)
    print("#" * 100)
    sys.stdout.flush()
    return response, price
