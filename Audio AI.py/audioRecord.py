from IPython.display import Javascript
from google.colab import output
from base64 import b64decode
from io import BytesIO
from pydub import AudioSegment

RECORD = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
    const reader = new FileReader();
    reader.onloadend = e => resolve(e.srcElement.result);
    reader.readAsDataURL(blob);
});

var record = time => new Promise(async resolve => {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.start();
    await sleep(time);
    recorder.onstop = async () => {
        blob = new Blob(chunks);
        text = await b2text(blob);
        resolve(text);
    };
    recorder.stop();
});
"""

def record(sec=3):
    display(Javascript(RECORD))
    s = output.eval_js('record(%d)' % (sec * 1000))
    b = b64decode(s.split(',')[1])
    audio = AudioSegment.from_file(BytesIO(b))
    return audio

def record_and_save(sec=3):
    display(Javascript(RECORD))
    s = output.eval_js('record(%d)' % (sec * 1000))
    b = b64decode(s.split(',')[1])
    with open('audio.wav', 'wb') as f:
        f.write(b)
    return 'audio.wav'

record_and_save()