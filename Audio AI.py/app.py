# audio to text
# from dotenv import find_dotenv, load_dot_env
from transformers import pipeline
from huggingface_hub import InferenceClient
import torch
import sysconfig; print(sysconfig.get_paths()["purelib"])
import _socket
def audioToText(url):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(torch.cuda.get_device_name(device))
    data, samplerate = sf.read(url) 
    print("samplerate: ",samplerate);
    print("Default device:", device)
    pipe = pipeline("automatic-speech-recognition", model="alvanlii/distil-whisper-small-cantonese",device=device)
    text = pipe(data)
    return text

result =audioToText("test.mp3");
print(result)
# load_dotenv(find_dotenv())

# def llm():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print(torch.cuda.get_device_name(device))
#     print("Default device:", device)
#     messages = [
#     {"role": "user", "content": "Who are you?"},]
#     pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True,device=device)
#     pipe(messages)

# llm();

# llm

#text to audio