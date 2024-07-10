# audio to text
# from dotenv import find_dotenv, load_dot_env
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,BitsAndBytesConfig,pipeline
import torch
from huggingface_hub import InferenceClient,login
import bitsandbytes
import numpy
print(numpy.__version__)
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
    print(transformers.__version__)
    print(bitsandbytes.__version__)
    print(numpy.__version__)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    #load the model
    pipe = pipeline("text-generation", model=model,model_kwargs={"torch_dtype": torch.bfloat16})
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