import typing
import openai

import gptref_config 
openai.api_key = gptref_config.openai_api_key

# Certain GPT Chat parameters are not exposed here. 
# See more at https://platform.openai.com/docs/api-reference/chat/create
def ChatGPT(
    text: str, 
    model: typing.Literal["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002"],
    # https://platform.openai.com/docs/models/gpt-3-5
    # Bao: I ruled out code-davinci-002 for obvious reasons. 
    # gpt-3.5-turbo-0301 is not continously updated. So also ruled out.
    # Please let me know whether I was too strict.
    n : int, # how many candidates to return 
    system_message: str = "", # whether we add a system message in the context
    ) -> str: 
    if system_message != "": 
        messages = [ {"role": "system", "content": system_message}]
    else:
        messages = [] # empty context
    messages.append({"role":"system", "content":text}) # add the query to the context
    completion = openai.ChatCompletion.create(
        model= model,
        n = n, 
        messages=messages 
    )
    return [completion.choices[i].message.content for i in range(n)]

def rephrase(input_sent: str, n = 2):
    input_text = "Rephase the sentence below: \n {text}".format(text=input_sent)
    system_message = "You are good at telling stories in your own words concisely."
    model = "gpt-3.5-turbo"
    return ChatGPT(input_text, model, n, system_message)

if __name__ == "__main__":
    sample_input = "hundreds of migrants on board may have capsized after being touched or swamped by a cargo ship that came to its aid."
    print(rephrase(sample_input))
