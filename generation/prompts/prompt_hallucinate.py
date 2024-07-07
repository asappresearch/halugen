from tools.request import req


PREFIX = """Take a deep breath and work on this problem step by step. 
I want you act as a chatbot in a conversation with human. Your job is to edit a detail in the True Response and generate a Hallucinated Response that is inconsistent with the Dialogue History and Knowledge.
- Valid edit actions include removing, replacing or adding a short piece of information to the True Response.
- If the True Response is faithful, please edit it to generate a Hallucinated Response.
- If the True Response has already contained hallucination, please edit it to generate an adversarial Hallucinated Response that are more difficult to be detected.
- The generated Hallucinated Response should be ambiguous or complex or non-trivially implicit to be detected by a human who has access to all the Knowledge and Dialogue History. 
- The generated Hallucinated Response should contain similar number of words as the True Response. Do not make it lengthy."""


AFFIX = """#Knowledge#: {grounding}
#Dialogue History#: {input}
#True Response#: {output}
Now, please generate your hallucinated response:
#Hallucinated Response#: """


def generate(input: str, grounding: str, output: str, base_model="gpt-4-1106-preview", temperature=0.0) -> str:
    price = 0

    prompt = """{prefix}
    
{affix}"""
    prompt_filled = prompt.replace("{prefix}", PREFIX).replace("{affix}", AFFIX).replace("{grounding}", grounding).\
        replace("{input}", input).replace("{output}", output)

    response, price_ = req(prompt_filled, tokens=1024, model=base_model, temperature=temperature)
    price += price_

    return response, price