# Qwen3 金融大模型微调、评测与部署
本项目基于通义千问 **Qwen3-4B-Thinking-2507** 大模型，实现金融领域的LoRA轻量化微调、本地交互式部署、模型性能自动化评测、Alpha Vantage实时股票数据对接，同时提供开箱即用的Web选股分析平台。  

作者：澳门城市大学 数据科学学院 智能科技服务专业 2022届 李硕彦

##  目录
- [环境要求](#环境要求)
- [文件清单与功能说明](#文件清单与功能说明)
- [快速开始](#快速开始)
  - [1. 环境安装](#1-环境安装)
  - [2. 模型微调](#2-模型微调)
  - [3. 微调后模型部署](#3-微调后模型部署)
  - [4. 模型性能评测](#4-模型性能评测)
  - [5. 实时股票数据对接MCP服务](#5-实时股票数据对接mcp服务)
  - [6. Web选股分析平台启动](#6-web选股分析平台启动)
- [参数配置](#参数配置)
- [免责声明](#免责声明)
- [许可证](#许可证)

##  环境要求
- 推荐Python 3.10 ~ 3.12 版本
- 推荐CUDA 11.8 及以上
- 此版本为针对CUDA环境下的Nvidia显卡进行部署，无法部署在AMD ROCm环境中
- 建议显存 < 8G 的显卡开启量化加载

##  文件清单与功能说明
| 文件名                  | 核心功能                                                                 | 关键依赖与注意事项                                                                 |
|-------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `train.py`              | Qwen3模型金融领域LoRA微调训练脚本                                       | 微调数据集：[Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) |
| `Qwen3_LoRA.py`         | 微调后LoRA模型的本地命令行交互式部署脚本                                 | 必须将训练输出的 `qwen3_4b_finance_lora` 文件夹放在与该脚本同一目录下           |
| `Qwen3_LoRA_TEST.py`    | 微调后模型的金融领域性能自动化评测脚本                                   | 测试数据集：[CFLUE金融评测集](https://www.modelscope.cn/datasets/tongyi_dianjin/CFLUE) |
| `Qwen3_LoRA_MCP.py`     | 对接Alpha Vantage MCP服务，获取实时股票/金融市场数据                     | 1. 先前往[Alpha Vantage MCP官网](https://mcp.alphavantage.co/)获取API密钥；<br>2. 密钥需写入同目录 `API.env` 文件 |
| `Qwen3_LoRA_Web.py`     | Web选股分析平台的Flask后端服务，提供模型推理与马科维茨有效前沿绘图能力   | 必须与 `web.html` 放在同一目录下，默认服务地址：`http://127.0.0.1:5000`        |
| `web.html`              | Web选股分析平台的前端页面代码                                             | 与 `Qwen3_LoRA_Web.py` 配套使用，不可单独运行                                     |

##  快速开始
### 1. 环境安装
1. 克隆/下载本项目到本地，进入项目根目录
2. 安装项目所需的全部Python依赖：
```bash
# 核心依赖安装
pip install torch transformers peft datasets accelerate python-dotenv requests flask matplotlib pandas numpy tqdm sentencepiece

# 4若显存 < 8G，建议开启量化。
# 实测在5070Ti Laptop下无量化加载，显存占用为8.7G ~ 9.8G
pip install bitsandbytes
```

### 2. 模型微调
1. 下载微调数据集到项目根目录，或保持网络通畅自动获取数据集
2. 执行微调脚本，训练完成后会自动生成 qwen3_4b_finance_lora 权重文件夹：
```bash
python train.py
```

### 3. 微调后模型部署
1. 确保 qwen3_4b_finance_lora 文件夹与 Qwen3_LoRA.py 在同一目录
2. 执行部署
```bash
python Qwen3_LoRA.py
```

### 4. 模型性能评测

1. 下载 CFLUE 测试集（名为test.json）到项目根目录的 CFLUE 文件夹内，命名为 test.json
2. 确保 LoRA 权重文件夹路径正确，执行评测脚本：
```bash
python Qwen3_LoRA_TEST.py
```
3. 可设置EVAL_RATIO的值，0.1表示为跑10%的测试集，1.0为u跑100%的测试集
```python
EVAL_RATIO = 0.1
```
4. 通过设置USE_SHUFFLE选择是否打乱测试集数据进行随机抽取，True为打乱，Flase为直接按顺序截取前n条数据
```python
USE_SHUFFLE = False 
```
5. 可设置随机种子，使每次抽取的题目都不一样
```python
RANDOM_SEED = 42 
```

### 5. 实时股票数据对接MCP服务
1. 前往 Alpha Vantage MCP 官网 注册并获取免费 API 密钥
2. 在项目根目录新建 API.env 文件，写入以下内容：
```env
ALPHA_VANTAGE_API_KEY=你的API密钥
```
3. 执行脚本，即可获取实时股票数据并结合大模型完成分析：
```bash
python Qwen3_LoRA_MCP.py
```

### 6. Web选股分析平台启动
1. 确保 Qwen3_LoRA_Web.py 与 web.html 在同一目录
2. 执行后端服务脚本：
```bash
python Qwen3_LoRA_Web.py
```
3. 脚本会自动加载模型并打开浏览器，访问 http://127.0.0.1:5000 即可使用选股分析平台


## 参数配置
### 1. 微调超参数调整
在 train.py 中可修改以下核心参数优化训练效果：  
1. r: LoRA 秩，越大拟合能力越强，显存占用越高，默认 16  
2. lora_alpha: LoRA 缩放系数，默认 32  
3. learning_rate: 学习率，默认 2e-4  
4. num_train_epochs: 训练轮数，默认 3  
5. per_device_train_batch_size: 单卡批次大小

### 2. 推理参数调整
在所有推理相关脚本中，可修改以下生成参数：
1. temperature: 温度系数，越低输出越稳定，越高创造性越强，默认 0.7
2. top_p: 核采样参数，默认 0.9
3. max_new_tokens: 最大生成 token 长度，可根据需求调整

## 免责声明
本项目仅用于技术学习与研究用途，所有金融相关的分析、建议、投资组合计算均为模型生成，不构成任何投资建议，不承担任何因使用本项目内容进行投资操作而产生的盈亏责任。

## 许可证
本项目代码遵循 Apache 2.0 开源许可证  
微调数据集遵循原数据集 Finance-Instruct-500k 的开源协议  
基础模型遵循 Qwen3 模型的官方开源许可协议