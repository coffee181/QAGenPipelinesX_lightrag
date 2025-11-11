@echo off
echo 启动QA生成管道（本地模型版本）...
echo.
echo 请确保：
echo 1. Ollama服务正在运行 (ollama serve)
echo 2. deepseek-r1:32b模型已下载 (ollama pull deepseek-r1:32b)
echo.
pause
qa_gen_pipeline_local.exe
pause
