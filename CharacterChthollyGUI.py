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
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


#===============================================================================================================
# 在cmd下,在chrome所在文件夹位置，执行：
# chrome.exe --remote-debugging-port=9222
# 如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E，建议登录。
# 接下来运行程序即可，注意不要关chrome和cmd
#===============================================================================================================

class TextThread(QThread):
    text_signal = pyqtSignal(str)


    def __init__(self, msg):
        super().__init__()
        options = webdriver.ChromeOptions()
        options.debugger_address = "localhost:9222"
        self.driver = webdriver.Chrome(options=options)
        # cheat cloudflare
        with open('stealth.min.js') as f:
            js = f.read()
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": js
            })
        self.msg = msg

    def run(self):
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
            element = driver.find_element(By.XPATH, '(//div[@style="overflow-wrap: break-word;"])[last()-1]')
            return element.text

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

        send_message(driver=self.driver, message=self.msg)

        wait_chth(driver=self.driver)

        self.text_signal.emit(get_response(driver=self.driver))

class VoiceThread(QThread):
    voice_signal = pyqtSignal(str)

    def __init__(self, jaresp):
        super().__init__()
        self.jaresp =jaresp

    def run(self):
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

        generateSound("[JA]" + self.jaresp + "[JA]")
        self.voice_signal.emit('')

class MyWindow(QWidget):
    signal = pyqtSignal(str)
    def __init__(self):
        print('GUI is creating...')
        super().__init__()
        self.init_ui()
        self.situation_history = list()
        self.msg_history = []  # 用来存放消息
        print('Tips1:you\'d better use English to chat with Chtholly.')
        print('Tips2:you\'d better not minimize the window of chrome if Timeout error occur.')
        # GoogleTrans, god bless you.
        self.translator = Translator()
        print('Chtholly is ready! Now you can chat with her.')
        time.sleep(1)

    def init_ui(self):
        self.setWindowTitle('ChatChtholly')
        self.resize(1000, 200)

        # 创建一个整体布局器
        container = QVBoxLayout()

        self.situation = QLabel()
        self.situation.resize(800, 10)
        self.situation.setWordWrap(True)
        self.situation.setAlignment(Qt.AlignBottom)

        self.msg = QLabel()
        self.msg.resize(800, 400)
        # print(self.msg.frameSize())
        self.msg.setWordWrap(True)  # 自动换行
        self.msg.setAlignment(Qt.AlignTop)  # 靠上
        # self.msg.setStyleSheet("background-color: yellow; color: black;")

        # 创建一个滚动对象
        scroll = QScrollArea()
        scroll.setWidget(self.msg)

        # 创建垂直布局器，用来添加自动滚动条
        v_layout = QVBoxLayout()
        v_layout.addWidget(scroll)

        self.edit = QLineEdit()
        self.edit.resize(800,200)
        self.edit.setPlaceholderText('say something to Chtholly~')
        form_layout = QFormLayout()
        form_layout.addRow("You:", self.edit)


        # 创建水平布局器
        h_layout = QHBoxLayout()
        self.btn = QPushButton("发送", self)
        # 绑定按钮的点击，点击按钮则开始检测
        self.btn.clicked.connect(self.check)
        h_layout.addStretch(1)  # 伸缩器
        h_layout.addWidget(self.btn)
        h_layout.addStretch(1)

        # 操作将要显示的控件以及子布局器添加到container
        container.addWidget(self.situation)
        container.addLayout(form_layout)
        container.addLayout(v_layout)
        container.addLayout(h_layout)

        # 设置布局器
        self.setLayout(container)

    def my_slot(self, msg):
        def wrap(self, string, max_width):
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

        self.textthread.terminate()

        # 更新内容
        self.msg_history.append('Chtholly:\n')
        self.msg.setText(str(self.msg_history))
        self.msg.resize(800, self.msg.frameSize().height() + 50)
        self.msg.repaint()

        self.msg_history.append(msg + '\n')
        self.msg.setText("".join(self.msg_history))
        self.msg.resize(800, self.msg.frameSize().height() + 400)
        self.msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

        trans = self.translator.translate(msg, dest='ja')
        trans_text = trans.text
        jaresp = word_process(str(trans_text))

        self.msg_history.append(jaresp + '\n')
        self.msg.setText("".join(self.msg_history))
        self.msg.resize(800, self.msg.frameSize().height() + 400)
        self.msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

        jaresp = flat(jaresp)

        self.situation.setText('Voice Generating...')
        self.msg.repaint()

        self.voicethread = VoiceThread(jaresp)
        self.voicethread.voice_signal.connect(self.final)
        self.voicethread.start()

    def final(self):
        self.voicethread.terminate()

        self.situation.setText('Successfully Generated!')
        self.situation.repaint()
        self.btn.setEnabled(True)
        PlaySound(r'.\output.wav', flags=1)


    def check(self):
        
        self.btn.setEnabled(False)
        self.situation.setText('Text Generating...')
        self.situation.repaint()
        self.textthread = TextThread(self.edit.text())
        self.textthread.text_signal.connect(self.my_slot)

        self.edit.clear()
        self.textthread.start()

############################################################################################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
