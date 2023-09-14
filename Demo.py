import commons
import sys
import re
import utils

from googletrans import Translator
from scipy.io.wavfile import write
from text import text_to_sequence
from models import SynthesizerTrn
from torch import no_grad, LongTensor
from characterai import PyCAI

TOKEN = '33a5650565d3ed003cd3af81e6be01efdd523022'
CHAR = 'LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E'

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def generateSound(inputString):
    model = r"./model/Chtholly.pth"
    config = r"./model/config.json"

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        if not emotion_embedding:
            text = inputString
            length_scale, text = get_label_value(
                text, 'LENGTH', 1, 'length scale')
            noise_scale, text = get_label_value(
                text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = get_label_value(
                text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = get_label(text, 'CLEANED')

            stn_tst = get_text(text, hps_ms, cleaned=cleaned)
            speaker_id = 0
            out_path = "output.wav"

            with no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)])
                sid = LongTensor([speaker_id])
                audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                       noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                    0, 0].data.cpu().float().numpy()

            write(out_path, hps_ms.data.sampling_rate, audio)
            print('Successfully saved!\n')


def wrap(string, max_width):
    result1 = [string[i:i + max_width] for i in range(0, len(string), max_width)]
    result = '\n'.join(result1)
    return result


def flatten(string):
    result1 = string.replace('\n', '')
    result = result1.replace(' ', '')
    return result

def preprocess(string):
    result = string.replace('\n', '')
    result = result.replace('#', '')
    result = result.replace('*', '')
    return result

def word_process(string):
    result = string.replace('Chtholly', 'クトリ')
    result = result.replace('Willem', 'ヴィレム')
    result = result.replace('William', 'ヴィレム')
    result = result.replace('nota', 'ノタ')
    result = result.replace('Nota', 'ノタ')
    result = result.replace('Seniorious', 'セニオリス')
    return result


if __name__ == "__main__":
    print('Tips1: You can use English or Chinese to chat with Chtholly.')
    print('Tips2: Chtholly\'s reply will be shown in both Chinese and Japanese, and spoken in Japanese.')

    # an unofficial api to c.ai
    client = PyCAI(TOKEN)
    client.start()
    chat = client.chat.get_chat(CHAR)
    participants = chat['participants']
    if not participants[0]['is_human']:
        tgt = participants[0]['user']['username']
    else:
        tgt = participants[1]['user']['username']
    # google trans, god bless you.
    translator = Translator()

    print('Chtholly is ready! Now you can chat with her.\n')

    if True:  # there used to be a loop
        message = input('You: ')

        print('\nText Generating...\n')

        data = client.chat.send_message(chat['external_id'], tgt, message)
        name = data['src_char']['participant']['name']
        text = data['replies'][0]['text']
        print(f"{name}: {text}")

        response = preprocess(str(text))
        trans = translator.translate(response, dest='ja')
        trans_text = trans.text
        jaresp = word_process(str(trans_text))
        print(jaresp)

        print('\nVoice Generating...')

        jaresp = flatten(jaresp)
        generateSound("[JA]" + jaresp + "[JA]")



