# 本地模型部署包使用说明

## 🎯 概述
这是支持本地模型的QA生成管道可执行文件，使用deepseek-r1:32b模型进行问答生成。

## 📋 使用前准备

### 1. 安装Ollama
```bash
# Windows: 下载并安装 https://ollama.ai/download
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. 启动Ollama服务
```bash
ollama serve
```

### 3. 下载模型
```bash
ollama pull deepseek-r1:32b
```

### 4. 测试模型
```bash
ollama run deepseek-r1:32b "你好，请介绍一下你自己"
```

## 🚀 使用方法

### 1. 配置环境
```bash
# 复制环境变量示例
copy .env.example .env

# 编辑.env文件（可选，使用默认配置即可）
```

### 2. 运行程序
```bash
# Windows
qa_gen_pipeline_local.exe

# Linux/macOS
./qa_gen_pipeline_local
```

## ⚙️ 配置说明

### config.yaml配置
程序会自动使用本地模型配置：
```yaml
question_generator:
  provider: "local"
  local:
    model_name: "deepseek-r1:32b"
    base_url: "http://localhost:11434"
    max_tokens: 2048
    temperature: 0.7
    timeout: 120
    questions_per_chunk: 30
```

### 切换回API模式
如需使用API模式，修改config.yaml：
```yaml
question_generator:
  provider: "deepseek"  # 改为deepseek使用API
```

## 🔧 故障排除

### 1. 模型连接失败
- 检查Ollama服务是否运行：`ollama serve`
- 检查模型是否下载：`ollama list`
- 检查端口是否被占用：`netstat -an | grep 11434`

### 2. GPU内存不足
- 使用更小的模型：`ollama pull deepseek-r1:7b`
- 检查GPU状态：`nvidia-smi`

### 3. 程序运行缓慢
- 检查GPU利用率：`nvidia-smi`
- 调整超时时间：修改config.yaml中的timeout值

## 💡 优势

- ✅ 完全免费，无API费用
- ✅ 数据安全，不离开本地
- ✅ 响应速度快，无网络延迟
- ✅ 无使用限制
- ✅ 完全离线运行

## 📞 技术支持

如遇问题，请检查：
1. Ollama服务状态
2. 模型下载情况
3. GPU内存使用
4. 配置文件设置

享受免费的本地AI服务！🎉
