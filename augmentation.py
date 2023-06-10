import typing
import openai
import time
import gptref_config 
openai.api_key = gptref_config.openai_api_key

# Certain GPT Chat parameters are not exposed here. 
# See more 'at https://platform.openai.com/docs/api-reference/chat/create

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def ChatGPT(
    texts: typing.List[str], 
    model: typing.Literal["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002"],
    # https://platform.openai.com/docs/models/gpt-3-5
    # Bao: I ruled out code-davinci-002 for obvious reasons. 
    # gpt-3.5-turbo-0301 is not continously updated. So also ruled out.
    # Please let me know whether I was too strict.
    n : int, # how many candidates to return 
    ) -> str: 
    completion = completion_with_backoff(
        model= model,
        n = n, 
        prompt=texts,
        max_tokens=gptref_config.MAX_NEW_TOKEN
    )
    num_results = len(completion.choices)
    assert num_results % n == 0
    placeholder = [None] * num_results
    # Align index
    for i in range(num_results): 
        placeholder[completion.choices[i].index] = completion.choices[i].text.strip()
    results = [[placeholder[i] for i in range(k*n, k*n+n)] for k in range(len(texts))]
    return results

def batched_request(input_text, **kwargs):
    total_samples = len(input_text)
    boundaries = list(range(0, total_samples, gptref_config.BATCH_SIZE))
    boundaries.append(total_samples)
    boundaries = list(zip(boundaries[:-1], boundaries[1:]))
    results = []
    for load_start, load_end in boundaries:
        batched_input = input_text[load_start:load_end]
        results.extend(ChatGPT(batched_input, **kwargs))
        time.sleep(60.0 / gptref_config.RPM)
    return results

def rephrase(input_sents: typing.List[str], n = 2):
    prompt = "Rephrase the sentence below: \n {text}"
    formated = [prompt.format(text=sent) for sent in input_sents]
    model = "text-babbage-001"
    return batched_request(formated, model=model, n=n)

def simplify(input_sents: typing.List[str], n = 2):
    prompt = "Simplify the sentence below: \n {text}"
    formated = [prompt.format(text=sent) for sent in input_sents]
    model = "text-davinci-003"
    return batched_request(formated, model=model, n=n)

if __name__ == "__main__":
    # sample_input1 = "hundreds of migrants on board may have capsized after being touched or swamped by a cargo ship that came to its aid."
    # sample_input2 = "Scientists believe a vaccine for Covid-19 might not be ready this year."
    # sents2aug = [sample_input1, sample_input2]
    sents2aug = ['This is the terrifying moment a bus driver rams into a stream of traffic as he falls asleep behind the wheel.', 
                 "The driver's side of the windscreen immediately shatters and falls through.", 
                 "driver's side were immediately shatters and falls through."]
    result = rephrase(sents2aug)
    print(result)
    # [
    #   ['A cargo ship that came to its aid may have swamped or touched hundreds of migrants on board, resulting in its capsizing.', 
    #   'A cargo ship that came to the aid of hundreds of migrants on board may have caused it to capsize after being touched or swamped.'], 
    #   ['It is unlikely that a vaccine for Covid-19 will be ready in 2020, according to scientists.', 
    #   'It is believed by scientists that a vaccine for Covid-19 might not be ready in 2020.']]