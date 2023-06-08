from functools import cache
import typing
import openai

import gptref_config 
openai.api_key = gptref_config.openai_api_key

# Certain GPT Chat parameters are not exposed here. 
# See more at https://platform.openai.com/docs/api-reference/chat/create
def ChatGPT(
    text: typing.List[str], 
    model: typing.Literal["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002"],
    # https://platform.openai.com/docs/models/gpt-3-5
    # Bao: I ruled out code-davinci-002 for obvious reasons. 
    # gpt-3.5-turbo-0301 is not continously updated. So also ruled out.
    # Please let me know whether I was too strict.
    n : int, # how many candidates to return 
    ) -> str: 
    completion = openai.Completion.create(
        model= model,
        n = n, 
        prompt=text,
        max_tokens=100
    )
    return completion

def rephrase(input_sents: typing.List[str], n = 2):
    prompt = "Rephase the sentence below: \n {text}"
    batched_input = [prompt.format(text=sent) for sent in input_sents]
    model = "text-davinci-003"
    return ChatGPT(batched_input, model, n)

if __name__ == "__main__":
    sample_input1 = "hundreds of migrants on board may have capsized after being touched or swamped by a cargo ship that came to its aid."
    sample_input2 = "Scientists believe a vaccine for Covid-19 might not be ready this year."
    result = rephrase([sample_input1, sample_input2])
