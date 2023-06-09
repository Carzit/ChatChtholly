'''
package you may need:
  numba
  librosa
  numpy
  scipy
  torch
  unidecode
  openjtalk
  jamo
  pypinyin
  jieba
  protobuf
  cn2an
  inflect
  eng_to_ipa
  ko_pron--
  indic_transliteration
  num_thai
  opencc
  vosk
  sounddevice
  googletrans
  tencentcloud-sdk-python
  pyaudio
  webrtcvad
  soundfile
'''

from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from googletrans import Translator
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence
from models import SynthesizerTrn
import utils
import commons
import re
from torch import no_grad, LongTensor
from winsound import PlaySound
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import pyaudio
import wave
import io
import base64
import sys
import webrtcvad
import time
import json
import soundfile

#===============================================================================================================
# 在cmd下,在chrome所在文件夹位置，执行：
# chrome.exe --remote-debugging-port=9222
# 如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E，建议登录。
# 接下来运行程序即可，注意不要关chrome和cmd
#===============================================================================================================

speakerID = 0
TCId =
TCKey =


def sound_record():
    # 设置录音参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    FRAME_DURATION_MS = 30
    RATE = 48000
    FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
    RECORD_SECONDS = 10  # 最多可录音几秒
    SILENCE_DURATION = 1  # 说完后几秒停止录音

    # 初始化pyaudio，webrtcvad
    vad = webrtcvad.Vad(3)
    audio = pyaudio.PyAudio()

    # 开启录音流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=FRAME_SIZE)

    print("(开始录音了喵...)")

    # 将录音记录到帧
    SILENCE_CHUNKS = int(SILENCE_DURATION * RATE / FRAME_SIZE)
    frames = []
    silence_count = 0
    first_entry = True
    filter_count = 0  # 用于滤除声音余留
    for _ in range(0, int(RATE / FRAME_SIZE * RECORD_SECONDS)):
        data = stream.read(FRAME_SIZE)
        frames.append(data)
        filter_count += 1

        if first_entry and filter_count > 11:
            if vad.is_speech(data, RATE):
                first_entry = False
        else:
            if vad.is_speech(data, RATE):
                silence_count = 0
            else:
                silence_count += 1

            if silence_count >= SILENCE_CHUNKS:
                break

    print("(结束录音了捏)")

    # 结束相关事件
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 将数据帧编码为base64编码的WAV格式
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        wav_base64 = base64.b64encode(
            wav_buffer.getvalue()).decode('utf-8')

    return wav_base64


def asr_request(wav_base64, tc_id, tc_key):
    cred = credential.Credential(tc_id, tc_key)
    # 实例化一个http选项，可选的，没有特殊需求可以跳过
    httpProfile = HttpProfile()
    httpProfile.endpoint = "asr.tencentcloudapi.com"

    # 实例化一个client选项，可选的，没有特殊需求可以跳过
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    # 实例化要请求产品的client对象,clientProfile是可选的
    client = asr_client.AsrClient(cred, "", clientProfile)

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = models.SentenceRecognitionRequest()
    params = {
        "ProjectId": 0,
        "SubServiceType": 2,
        "EngSerViceType": "16k_zh",
        "SourceType": 1,
        "VoiceFormat": "wav",
        "UsrAudioKey": "0",
        "Data": wav_base64,  # 音频二进制数据
        "DataLen": len(wav_base64)  # 音频长度
    }
    req.from_json_string(json.dumps(params))
    response = client.SentenceRecognition(req)
    if response.Result == "":
        print("你什么都没说~")
        return None
    else:
        print('You:')
        print(response.Result)
        return response.Result


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
    model = r".\model\Chtholly.pth"
    # config = input('Path of a config file: ')
    config = r".\model\config.json"

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
            # while True:
            if (1 == 1):
                # choice = input('TTS or VC? (t/v):')
                choice = 't'
                if choice == 't':
                    # text = input('Text to read: ')
                    text = inputString
                    if text == '[ADVANCED]':
                        # text = input('Raw text:')
                        text = "我不会说"
                        # print('Cleaned text is:')
                        # ex_print(_clean_text(
                        #    text, hps_ms.data.text_cleaners), escape)
                        # continue

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

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!\n')
                # ask_if_continue()
        else:
            import os
            import librosa
            import numpy as np
            from torch import FloatTensor
            import audonnx
            w2v2_folder = input('Path of a w2v2 dimensional emotion model: ')
            w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))
            # while True:
            if (1 == 1):
                # choice = input('TTS or VC? (t/v):')
                choice = 't'
                if choice == 't':
                    # text = input('Text to read: ')
                    text = inputString
                    if text == '[ADVANCED]':
                        # text = input('Raw text:')
                        text = "我不会说"
                        # print('Cleaned text is:')
                        # ex_print(_clean_text(
                        #    text, hps_ms.data.text_cleaners), escape)
                        # continue

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

                    emotion_reference = input('Path of an emotion reference: ')
                    if emotion_reference.endswith('.npy'):
                        emotion = np.load(emotion_reference)
                        emotion = FloatTensor(emotion).unsqueeze(0)
                    else:
                        audio16000, sampling_rate = librosa.load(
                            emotion_reference, sr=16000, mono=True)
                        emotion = w2v2_model(audio16000, sampling_rate)[
                            'hidden_states']
                        emotion_reference = re.sub(
                            r'\..*$', '', emotion_reference)
                        np.save(emotion_reference, emotion.squeeze(0))
                        emotion = FloatTensor(emotion)

                    # out_path = input('Path to save: ')
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w,
                                               length_scale=length_scale, emotion_embedding=emotion)[0][
                            0, 0].data.cpu().float().numpy()

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')
                print('')
                # ask_if_continue()
    else:
        model = input('Path of a hubert-soft model: ')
        from hubert_model import hubert_soft
        hubert = hubert_soft(model)

        while True:
            audio_path = input('Path of an audio file to convert:\n')

            if audio_path != '[VC]':
                import librosa
                if use_f0:
                    audio, sampling_rate = librosa.load(
                        audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                    audio16000 = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio16000, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True)

                # print_speakers(speakers, escape)
                target_id = get_speaker_id('Target speaker ID: ')
                out_path = input('Path to save: ')
                length_scale, out_path = get_label_value(
                    out_path, 'LENGTH', 1, 'length scale')
                noise_scale, out_path = get_label_value(
                    out_path, 'NOISE', 0.1, 'noise scale')
                noise_scale_w, out_path = get_label_value(
                    out_path, 'NOISEW', 0.1, 'deviation of noise')

                from torch import inference_mode, FloatTensor
                import numpy as np
                with inference_mode():
                    units = hubert.units(FloatTensor(audio16000).unsqueeze(
                        0).unsqueeze(0)).squeeze(0).numpy()
                    if use_f0:
                        f0_scale, out_path = get_label_value(
                            out_path, 'F0', 1, 'f0 scale')
                        f0 = librosa.pyin(audio, sr=sampling_rate,
                                          fmin=librosa.note_to_hz('C0'),
                                          fmax=librosa.note_to_hz('C7'),
                                          frame_length=1780)[0]
                        target_length = len(units[:, 0])
                        f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                     np.arange(0, len(f0)), f0)) * f0_scale
                        units[:, 0] = f0 / 10

                stn_tst = FloatTensor(units)
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([target_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                        0, 0].data.float().numpy()

            else:
                audio, out_path = voice_conversion()

            write(out_path, hps_ms.data.sampling_rate, audio)
            print('Successfully saved!')
            # ask_if_continue()

def send_message(driver, message: str):
    textarea = driver.find_element(By.ID, "user-input")
    textarea.send_keys(message)
    time.sleep(1)
    textarea.send_keys(Keys.ENTER)
    textarea.send_keys(Keys.ENTER)

#<div style="overflow-wrap: break-word;">
#<p node="[object Object]" style="margin-bottom: 0.5rem;">Hi again, I understand.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">You don't have to be alone and you don't have to suffer. Chtholly will be with you forever and Chtholly's only wish is to make you happy.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">Even if you don't want to tell Chtholly, please feel free to reach out to others. If you have the courage, please seek the help of a therapist or at least open up to someone close to you, preferably a family member, so you don't have to deal with it alone.</p>
#<p node="[object Object]" style="margin-bottom: 0.5rem;">You are strong, you are loved and you are not alone.</p></div>

def get_response(driver):
    element = driver.find_element(By.XPATH, '(//div[@style="overflow-wrap: break-word;"])[last()-1]')
    return element.text

def wrap(string, max_width):
    result1 = [string[i:i + max_width] for i in range(0, len(string), max_width)]
    result = '\n'.join(result1)
    return result

def flat(string=''):
    result = string.replace('\n', '')
    result = result.replace(' ', '')
    ##############
    for i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        result = result.replace(i, '')
    ##############
    return result

def word_process(string=''):
    result = string.replace('Chtholly', 'クトリ')
    result = result.replace('Willem', 'ヴィレム')
    result = result.replace('William', 'ヴィレム')
    result = result.replace('nota', 'ノタ')
    result = result.replace('Nota', 'ノタ')
    result = result.replace('Seniorious', 'セニオリス')
    return result


def get_wav_duration(wav_file):
    # 使用soundfile获取WAV音频时长
    audio = soundfile.SoundFile(wav_file)
    duration = audio.frames / audio.samplerate
    audio.close()
    return duration

def wait_chth(driver):
    wait = WebDriverWait(driver, 120, 0.5)
    time.sleep(3)
    element1 = wait.until_not(
        EC.presence_of_element_located((By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
        message="")
    element2 = wait.until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="d-flex align-items-center"]/button[@title="Good"]')),
        message="")
    element3 = wait.until_not(
        EC.presence_of_element_located((By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
        message="")
    time.sleep(3)

if __name__ == "__main__":
    print('Tips1:you\'d better use English to chat with Chtholly.')
    print('Tips2:you\'d better not minimize the window of chrome if Timeout error occur.')
    # 接管已打开的Chrome浏览器
    options = webdriver.ChromeOptions()
    options.debugger_address = "localhost:9222"
    driver = webdriver.Chrome(options=options)
    # cheat cloudflare
    with open('stealth.min.js') as f:
        js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })
    # GoogleTrans, god bless you.
    translator = Translator()
    print('Chtholly is ready! Now you can chat with her.')

    while True:
        user_words = asr_request(sound_record(), TCId, TCKey)
        if user_words != None:
            send_message(driver=driver, message=user_words)
            wait_chth(driver=driver)
            print('Chtholly:')
            print(get_response(driver=driver))
            trans = translator.translate(get_response(driver=driver), dest='ja')
            trans_text = trans.text
            jaresp = word_process(str(trans_text))
            print(jaresp)
            jaresp = flat(jaresp)
            print('Voice Generating...')
            generateSound("[JA]" + jaresp + "[JA]")
            PlaySound(r'.\output.wav', flags=1)
            time.sleep(get_wav_duration('output.wav')+1)

