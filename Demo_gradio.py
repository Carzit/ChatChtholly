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
from winsound import PlaySound

import gradio as gr

def respond(message, history):
    speakerID = 0

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
        # model = input('Path of a VITS model: ')
        model = r"./model/Chtholly.pth"
        # config = input('Path of a config file: ')
        config = r"./model/config.json"

        hps_ms = utils.get_hparams_from_file(config)
        n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
        speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
        use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
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
                # choice = input('TTS or VC? (t/v):')
                # choice = 't'
                # text = input('Text to read: ')
                text = inputString
                length_scale, text = get_label_value(
                    text, 'LENGTH', 1, 'length scale')
                noise_scale, text = get_label_value(
                    text, 'NOISE', 0.667, 'noise scale')
                noise_scale_w, text = get_label_value(
                    text, 'NOISEW', 0.8, 'deviation of noise')
                cleaned, text = get_label(text, 'CLEANED')

                stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                # speaker_id = get_speaker_id('Speaker ID: ')
                speaker_id = speakerID
                # out_path = input('Path to save: ')
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

    def get_response(message):
        client = PyCAI('33a5650565d3ed003cd3af81e6be01efdd523022')
        translator = Translator()
        data = client.chat.send_message(char='LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E',
                                        token='33a5650565d3ed003cd3af81e6be01efdd523022',
                                        message=message,
                                        wait=True)
        response = preprocess(str(f"{data['replies'][0]['text']}"))

        trans = translator.translate(response, dest='ja')
        trans_text = trans.text
        jaresp = word_process(str(trans_text))

        flat_jaresp = flatten(jaresp)
        generateSound("[JA]" + flat_jaresp + "[JA]")

        response = response + '\n' + jaresp
        PlaySound(r'.\output.wav', flags=1)

        return response

    return str(get_response(message))


gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Say something to Chtholly", container=False, scale=7),
    title="ChatChtholly",
    description="To the happiest girl in the world",
    theme="soft",
    retry_btn=None,
    undo_btn="Delete",
    clear_btn="Clear",
    ).launch()