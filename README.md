# voice
可以部署在本地并提供接口的语音转文字和文字转语音¿

给AstrBot的，能用就行

我电脑不行，openai_adapter就不测了

vosk命令：uvicorn vosk_openai_api:app --host 0.0.0.0 --port 8000

gtts命令：uvicorn openai_adapter:app --host 0.0.0.0 --port 11559

小声BB：其实还藏了不少的（
