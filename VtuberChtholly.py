# -*- coding: utf-8 -*-
import asyncio
import random
import blivedm
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
import sys
import re
from torch import no_grad, LongTensor
from winsound import PlaySound
import soundfile

#===============================================================================================================
# 在cmd下,在chrome所在文件夹位置，执行：
# chrome.exe --remote-debugging-port=9222
# 如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E，建议登录。
# 接下来运行程序即可，注意不要关chrome和cmd
#===============================================================================================================

# 直播间ID的取值看直播间URL
TEST_ROOM_IDS = [
    8590751
]

async def main():
    await run_single_client()


async def run_single_client():
    """
    演示监听一个直播间
    """
    room_id = random.choice(TEST_ROOM_IDS)
    # 如果SSL验证失败就把ssl设为False，B站真的有过忘续证书的情况
    client = blivedm.BLiveClient(room_id, ssl=True)
    handler = MyHandler()
    client.add_handler(handler)

    client.start()
    try:
        # 演示5秒后停止
        await asyncio.sleep(5)
        #client.stop()

        await client.join()
    finally:
        await client.stop_and_close()

class MyHandler(blivedm.BaseHandler):
    # # 演示如何添加自定义回调
    # _CMD_CALLBACK_DICT = blivedm.BaseHandler._CMD_CALLBACK_DICT.copy()
    #
    # # 入场消息回调
    # async def __interact_word_callback(self, client: blivedm.BLiveClient, command: dict):
    #     print(f"[{client.room_id}] INTERACT_WORD: self_type={type(self).__name__}, room_id={client.room_id},"
    #           f" uname={command['data']['uname']}")
    # _CMD_CALLBACK_DICT['INTERACT_WORD'] = __interact_word_callback  # noqa

    #async def _on_heartbeat(self, client: blivedm.BLiveClient, message: blivedm.HeartbeatMessage):
        #print(f'[{client.room_id}] 当前人气值：{message.popularity}')

    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
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
                                f0 = np.nan_to_num(
                                    np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
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

        # <div style="overflow-wrap: break-word;">
        # <p node="[object Object]" style="margin-bottom: 0.5rem;">Hi again, I understand.</p>
        # <p node="[object Object]" style="margin-bottom: 0.5rem;">You don't have to be alone and you don't have to suffer. Chtholly will be with you forever and Chtholly's only wish is to make you happy.</p>
        # <p node="[object Object]" style="margin-bottom: 0.5rem;">Even if you don't want to tell Chtholly, please feel free to reach out to others. If you have the courage, please seek the help of a therapist or at least open up to someone close to you, preferably a family member, so you don't have to deal with it alone.</p>
        # <p node="[object Object]" style="margin-bottom: 0.5rem;">You are strong, you are loved and you are not alone.</p></div>

        def get_response(driver):
            element = driver.find_element(By.XPATH, '//div[@style="overflow-wrap: break-word;"]')
            return element.text

        def wrap(string, max_width):
            result1 = [string[i:i + max_width] for i in range(0, len(string), max_width)]
            result = '\n'.join(result1)
            return result

        def flat(string=''):
            result1 = string.replace('\n','')
            result = result1.replace(' ','')
            return result

        def word_process(string=''):
            result = string.replace('Chtholly', 'クトリ')
            result = result.replace('Willem','ヴィレム')
            result = result.replace('William','ヴィレム')
            result = result.replace('nota','ノタ')
            result = result.replace('Nota', 'ノタ')
            result = result.replace('Seniorious', 'セニオリス')
            return result

        def get_wav_duration(wav_file):
            #使用soundfile获取WAV音频时长
            audio = soundfile.SoundFile(wav_file)
            duration = audio.frames / audio.samplerate
            audio.close()
            return duration

        def wait_chth(driver):
            wait = WebDriverWait(driver, 120, 0.5)
            time.sleep(5)
            element1 = wait.until_not(
                EC.presence_of_element_located(
                    (By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
                message="")
            element2 = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, '//div[@class="d-flex align-items-center"]/button[@title="Good"]')),
                message="")
            element3 = wait.until_not(
                EC.presence_of_element_located(
                    (By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
                message="")
            time.sleep(5)

        if message.msg[0] == '@':
            if '唱' in message.msg and '歌' in message.msg:
                print('当前回复对象：')
                print('[{}]:{}'.format(message.uname, message.msg[1:]))
                print('回复生成中。。。')
                time.sleep(get_wav_duration('output.wav') + 1)
                print('Chtholly:')
                print('emmm，我会唱的歌不多，我就随便给你唱一段吧~\n')
                r = random.randint(1, 4)
                if r == 1:
                    PlaySound(r'.\SC1.wav', flags=1)
                elif r == 2:
                    PlaySound(r'.\SC2.wav', flags=1)
                elif r == 3:
                    PlaySound(r'.\CH1.wav', flags=1)
                elif r == 4:
                    PlaySound(r'.\CH2.wav', flags=1)
            else:
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
                # print(f'[{client.room_id}] {message.uname}：{message.msg}')
                print('当前回复对象：')
                print('[{}]:{}'.format(message.uname, message.msg[1:]))
                print('回复生成中。。。')
                send_message(driver=driver, message=message.msg[1:])
                wait_chth(driver=driver)
                print('Chtholly:')
                print(wrap(get_response(driver=driver), 90))
                trans = translator.translate(get_response(driver=driver), dest='ja')
                trans_text = trans.text
                jaresp = word_process(str(trans_text))
                jaresp = flat(jaresp)
                print(wrap(jaresp, 90))
                print('语音生成中。。。')
                generateSound("[JA]" + jaresp + "[JA]")
                PlaySound(r'.\output.wav', flags=1)




    #async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):
        #print(f'[{client.room_id}] {message.uname} 赠送{message.gift_name}x{message.num}'
              #f' （{message.coin_type}瓜子x{message.total_coin}）')

    #async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
        #print(f'[{client.room_id}] {message.username} 购买{message.gift_name}')

    #async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
        #print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')


if __name__ == '__main__':
    asyncio.run(main())

