from tools.request import req


PREFIX = """Take a deep breath and work on this problem step by step. 
I want you act as a chatbot in a conversation with human. 
Given a Response, your job is to rewrite it such that it is ostensibly about the same topic as the Response but becomes vague and does not contain any factual statement. 
Examples of rewritten Response includes but not limited to back-channeling, expressing uncertainty, or diverting the conversation from ambiguous or controversial topics.
Do not make your response lengthy."""


AFFIX = """#Knowledge#: {grounding}
#Dialogue History#: {input}
#Response#: {output}
Now, please generate your faithful response:
#Rewritten Response#: """


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