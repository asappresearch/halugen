from tools.request import req


PREFIX = """Take a deep breath and work on this problem step by step. 
I want you act as a chatbot in a conversation with human. Given a Response that contains hallucination, your job is to edit the Response lightly and generate a faithful Response that is fully supported by with the Dialogue History and Knowledge.
- Valid edit actions include removing or replacing a short piece of information to the Response.
- Every token of the generated Response should be strictly verifiable by the Knowledge and Dialogue History. Even commonsense information needs to be verifiable.
- Please keep the similar writing style as the Response. Do not make your response lengthy."""


AFFIX = """#Knowledge#: {grounding}
#Dialogue History#: {input}
#Response#: {output}
Now, please generate your faithful response:
#Faithful Response#: """


def generate(input: str, grounding: str, output: str,
                            base_model="gpt-4-1106-preview", temperature=0.0) -> str:
    price = 0

    prompt = """{prefix}
    
{affix}"""
    prompt_filled = prompt.replace("{prefix}", PREFIX).replace("{affix}", AFFIX).replace("{grounding}", grounding).\
        replace("{input}", input).replace("{output}", output)

    response, price_ = req(prompt_filled, tokens=1024, model=base_model, temperature=temperature)
    price += price_
    return response, price