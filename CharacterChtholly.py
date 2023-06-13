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
  pyChatGPT
  vosk
  sounddevice
  googletrans
'''

from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from googletrans import Translator
from scipy.io.wavfile import write
from text import text_to_sequence
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
from winsound import PlaySound

#===============================================================================================================
# 在cmd下,在chrome所在文件夹位置，执行：
# chrome.exe --remote-debugging-port=9222
# 如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E，建议登录。
# 接下来运行程序即可，注意不要关chrome和cmd
#===============================================================================================================

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
    result1 = string.replace('\n', '')
    result = result1.replace(' ', '')
    return result

def word_process(string=''):
    result = string.replace('Chtholly', 'クトリ')
    result = result.replace('Willem', 'ヴィレム')
    result = result.replace('William', 'ヴィレム')
    result = result.replace('nota', 'ノタ')
    result = result.replace('Nota', 'ノタ')
    result = result.replace('Seniorious', 'セニオリス')
    return result

def wait_chth(driver):
    wait = WebDriverWait(driver, 120, 0.5)
    time.sleep(5)
    element1 = wait.until_not(
        EC.presence_of_element_located((By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
        message="")
    element2 = wait.until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="d-flex align-items-center"]/button[@title="Good"]')),
        message="")
    element3 = wait.until_not(
        EC.presence_of_element_located((By.XPATH, '//div[@class="input-group me-3 my-0"]/svg[stroke="currentColor"]')),
        message="")
    time.sleep(5)

if __name__ == "__main__":
    print('Tips1:you\'d better use English to chat with Chtholly.')
    print('Tips2:you\'d better not minimize the window of chrome if Timeout error occur.')
    # 接管已打开的Chrome浏览器
    options = webdriver.ChromeOptions()
    # options.debugger_address = "localhost:9222"
    driver = webdriver.Chrome(options=options)
    driver.get(f'https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E')
    # cheat cloudflare
    with open('stealth.min.js') as f:
        js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })
    # cheat c.ai
    driver.execute_script("""
            var headers = new Headers();
            headers.append("Authorization", "Token " + "LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E");
            headers.append("Content-Type", "application/json");
        """)
    # GoogleTrans, god bless you.
    translator = Translator()
    print('Chtholly is ready! Now you can chat with her.')


    while True:
        send_message(driver=driver, message=input('You:\n'))
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