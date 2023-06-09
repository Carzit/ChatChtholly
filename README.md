# 简易黄金妖精召唤术 Simple Invocation of Repurakaan
[![Open Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Carzit/ChatChtholly/blob/main/ChatChthollyDemo.ipynb) 

### VITS模型下载：
您可通过Release下载: https://github.com/Carzit/ChatChtholly/releases/tag/MODEL  
或根据model文件夹下MODEL DOWNLOAD.md文件中的网盘地址: https://pan.baidu.com/s/1MtAm3TpjhqrmzJmS7eqIoQ?pwd=suka  
下载珂朵莉的VITS模型及对应config配置文件，并将其放置在model文件夹中。

## ChatChtholly（该程序由于openai近期的设置变更暂不可用，请使用CharacterChtholly!）
#### (Through Hypnotized ChatGPT)
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
 
## CharacterChtholly
#### (Through pretrained character model on c.ai)
1.1:安装anaconda环境或Python>=3.7，浏览器请使用Chrome  
1.2:pip安装项目所需要的库文件(见requirements.txt或主程序的注释）  
2.1:在cmd下,在chrome所在文件夹位置，执行：chrome.exe --remote-debugging-port=9222  
如此打开chrome后 ,手动访问https://beta.character.ai/chat?char=LMri6f9uZj2p17QoKDiEvDw1wAk2AUoi1C02V6HHU8E

*建议登录账号（否则有会话限制）  
*注意不要关闭chrome和cmd  

2.2:运行CharacterChtholly.py  
3.1:开始和Chtholly聊天叭  
当控制台提示"You:"时可以输入您想说的语句，回车后即发送到c.ai，珂朵莉的回答生成完毕后会显示在终端上，稍等几秒后对应的语音输出也会自动播放  

4.1:如果您想使用语音识别功能（您可以直接语音输入），可以使用CharacterChthollyVoice.py这个程序。您需要申请腾讯云的API，并将程序第69-70行的TC_ID与TC_KEY赋上您的语音识别api的ID与KEY值。

申请操作如下：  
https://cloud.tencent.com/ 打开腾讯云网站，点击右上角注册账号。  
注册后在 https://cloud.tencent.com/product/asr 打开腾讯云ASR服务，点击立即使用，选择新用户专享资源包，开通相关服务。   
开通服务后，在https://console.cloud.tencent.com/cam/capi 打开腾讯云控制台API密钥管理页面，选择新建密钥。  
创建成功后即可获取该密钥相应的ID与KEY值。  
 
### （我尚在优化这一项目） 

## 鸣谢
- [MoeGoe_GUI]https://github.com/CjangCjengh/MoeGoe_GUI
- [Pretrained models]https://github.com/CjangCjengh/TTSModels
- [PyChatGPT]https://github.com/terry3041/pyChatGPT
- [ChatWaifu]https://github.com/cjyaddone/ChatWaifu
- [Sukasuka-vocal-dataset-builder]https://github.com/Hecate2/sukasuka-vocal-dataset-builder
