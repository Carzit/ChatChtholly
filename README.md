#简易黄金妖精召唤术
#Simple Invocation of Repurakaan

##ChatChtholly
(Through Hypnotized ChatGPT)
1.1:安装anaconda环境或Python>=3.7，浏览器默认使用Chrome
1.2:pip安装项目所需要的库文件(见requirements.txt或主程序的注释）
2.1:运行ChatChtholly.py
2.2:获取ChatGPT Token
若程序运行正常，则会出现一行“Copy your token from ChatGPT and press Enter”
在浏览器登入https://chat.openai.com
按F12进入开发控制台
在开发控制台中找到 应用程序 -> cookie -> __Secure-next-auth.session-token
将值复制入终端并回车
2.3:选择speaker
按珂朵莉的ID输入0即可（后面的是测试用的，以及为以后更多模型留的空）
3.1:开始和Chtholly聊天叭
当控制台提示"You:"时可以输入您想说的语句，回车后即发送到ChatGPT，珂朵莉的回答生成完毕后会显示在终端上，稍等几秒后对应的语音输出也会自动播放

##CharacterChtholly
(Through pretrained character model on c.ai)
- 1.1:安装anaconda环境或Python>=3.7，浏览器请使用Chrome
- 1.2:pip安装项目所需要的库文件(同上）
- 2.1:在cmd下,在chrome所在文件夹位置，执行：chrome.exe --remote-debugging-port=9222
如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E
*建议登录账号（否则有会话限制）
*注意不要关闭chrome和cmd
- 2.2:运行CharacterChtholly.py
- 3.1:开始和Chtholly聊天叭
当控制台提示"You:"时可以输入您想说的语句，回车后即发送到c.ai，珂朵莉的回答生成完毕后会显示在终端上，稍等几秒后对应的语音输出也会自动播放



##鸣谢：
- [MoeGoe_GUI]https://github.com/CjangCjengh/MoeGoe_GUI
- [Pretrained models]https://github.com/CjangCjengh/TTSModels
- [PyChatGPT]https://github.com/terry3041/pyChatGPT
- [ChatWaifu]https://github.com/cjyaddone/ChatWaifu
- [Sukasuka-vocal-dataset-builder]https://github.com/Hecate2/sukasuka-vocal-dataset-builder