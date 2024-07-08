# audio to text
# from dotenv import find_dotenv, load_dot_env
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
import torch
from huggingface_hub import InferenceClient
from huggingface_hub import login
import torch

api_token ="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"

# Log in to the Hugging Face model hub
login(api_token)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.get_device_name(device))
print("Default device:", device)

def audioToText(url):
    pipe = pipeline("automatic-speech-recognition", model="alvanlii/distil-whisper-small-cantonese",device=device)
    text = pipe(url)
    return text

# ASRoutput =audioToText("test.mp3");
# print(ASRoutput)

def llm(messages):
    #load the model
    model = AutoModel.from_pretrained("backyardai/llama-3-8b-Instruct-GGUF")
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct",device_map="auto",model_kwargs={"torch_dtype": torch.bfloat16})
    #prompt template
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt = True,
    )
    print("prompt",prompt)
    #identify the end of text
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    torch.cuda.empty_cache()
    response = pipe(
        prompt,
        max_new_tokens= 256,
        eos_token_id = terminators,
        do_sample = True, # text generation stragaries
        temperature = 0.6,
        top_p = 0,
    )
    print(response)
    print("dddddddddd")
    print(response[0]["generated_text"][len(prompt):])
    return response

messages = [{"role": "user", "content": "Who are you?"}]
llm_output = llm(messages)
print(llm_output)
# llm

#text to audio