import commons
import sys
import re
import utils

from googletrans import Translator
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence
from models import SynthesizerTrn
from torch import no_grad, LongTensor
from characterai import PyCAI

speakerID = 0

def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


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
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False
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

    def voice_conversion():
        audio_path = input('Path of an audio file to convert:\n')
        print_speakers(speakers)
        audio = utils.load_audio_to_torch(
            audio_path, hps_ms.data.sampling_rate)

        originnal_id = get_speaker_id('Original speaker ID: ')
        target_id = get_speaker_id('Target speaker ID: ')
        out_path = input('Path to save: ')

        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, hps_ms.data.filter_length,
                                 hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                                 center=False)
        spec_lengths = LongTensor([spec.size(-1)])
        sid_src = LongTensor([originnal_id])

        with no_grad():
            sid_tgt = LongTensor([target_id])
            audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[
                0][0, 0].data.cpu().float().numpy()
        return audio, out_path

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

            # print_speakers(speakers, escape)
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
            # ask_if_continue()

#<div style="overflow-wrap: break-word;">
#<p node="[object Object]" style="margin-bottom: 0.5rem;">Hi again, I understand.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">You don't have to be alone and you don't have to suffer. Chtholly will be with you forever and Chtholly's only wish is to make you happy.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">Even if you don't want to tell Chtholly, please feel free to reach out to others. If you have the courage, please seek the help of a therapist or at least open up to someone close to you, preferably a family member, so you don't have to deal with it alone.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">You are strong, you are loved and you are not alone.</p></div>

def wrap(string, max_width):
    result1 = [string[i:i + max_width] for i in range(0, len(string), max_width)]
    result = '\n'.join(result1)
    return result

def flat(string=''):
    result1 = string.replace('\n', '')
    result = result1.replace(' ', '')
    return result

def preprocess(string = ''):
    result = string.replace('\n', '')
    result = result.replace('#', '')
    result = result.replace('*', '')
    return result

def word_process(string=''):
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
    client = PyCAI('33a5650565d3ed003cd3af81e6be01efdd523022')
    # GoogleTrans, god bless you.
    translator = Translator()
    print('Chtholly is ready! Now you can chat with her.\n')
    
    if True:# there used to be a loop
        message = input('You: \n')
        print('\nText Generating...\n')
        data = client.chat.send_message(char='LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E',
                                        token='33a5650565d3ed003cd3af81e6be01efdd523022',
                                        message=message,
                                        wait=True)
        response = preprocess(str(f"{data['replies'][0]['text']}"))
        print('Chtholly:')
        print(response)
        trans = translator.translate(response, dest='ja')
        trans_text = trans.text
        jaresp = word_process(str(trans_text))
        print(jaresp)
        jaresp = flat(jaresp)
        print('\nVoice Generating...')
        generateSound("[JA]" + jaresp + "[JA]")