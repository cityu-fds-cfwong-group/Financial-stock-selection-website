import os
import threading
import webbrowser
from flask import Flask, request, jsonify, send_from_directory

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  #使用国内镜像
os.environ["HF_OFFLINE"] = "1" #强制离线模式
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1" #禁用遥测，加快启动速度

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

## 第三部分新增 ##
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # 【重要修复】必须添加这个后端设置，否则Flask画图会报错导致500
import matplotlib.pyplot as plt

import io
import base64

# 初始化 Flask 应用
app = Flask(__name__)

# 全局变量存放模型，避免重复加载
global_model = None
global_tokenizer = None
script_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 1. 网页托管路由 =================
@app.route('/')
def index():
    # 直接将当前目录下的 筛选部.html 作为主页返回
    return send_from_directory(script_dir, 'web_1.html')

# ================= 2. 模型推理 API =================
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_input = data.get("prompt", "")
        if not user_input:
            return jsonify({"status": "error", "message": "未接收到股票数据"}), 400

        messages = [
            {"role": "system", "content": "你是一个专业的金融助手，擅长分析A股基本面数据并给出客观的投资建议。"},
            {"role": "user", "content": user_input}
        ]
        
        try:
            text = global_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>system\n你是一个专业的金融助手，擅长分析A股基本面数据并给出客观的投资建议。<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        inputs = global_tokenizer(text, return_tensors="pt").to(global_model.device)

        # 生成回答
        with torch.no_grad():
            outputs = global_model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=global_tokenizer.pad_token_id,
                eos_token_id=global_tokenizer.eos_token_id
            )

        # 解码输出
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = global_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return jsonify({"status": "success", "result": response})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# ================新增马科维兹曲线画图API==================
@app.route('/api/markowitz', methods=['POST'])
def markowitz_api():
    data = request.json
    # 自动识别筛选后的前三只股票
    top_stocks = data.get('stocks', [])[:3]
    
    if len(top_stocks) < 2:
        return jsonify({"error": "需要至少两只股票进行优化"}), 400
        
    try:
        plot_b64, weights = perform_markowitz_optimization(top_stocks)
        return jsonify({
            "plot": plot_b64,
            "weights": weights,
            "stocks_used": top_stocks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# --- 新增：Markowitz核心逻辑函数 ---
def perform_markowitz_optimization(stock_list):
    """
    模拟获取数据并计算马科维茨曲线
    注意：实际应用中这里应调用 yfinance 或 数据库获取真实历史收盘价
    """
    # 模拟历史收益率数据 (实际应根据 stock_list 抓取)
    np.random.seed(42)
    num_stocks = len(stock_list)
    returns = np.random.randn(100, num_stocks) * 0.05 + 0.001 
    
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)
    
    # 蒙特卡洛模拟
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # 计算投资组合收益和标准差
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = results[0,i] / results[1,i] # 夏普比率

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Markowitz Efficient Frontier - Top 3 Stocks')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Returns')
    
    # 获取夏普比率最高的点
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    plt.scatter(sdp, rp, marker='*', color='r', s=100, label='Maximum Sharpe ratio')
    plt.legend()

    # 将图片转为base64供前端显示
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # 获取最优权重
    best_weights = weights_record[max_sharpe_idx]
    weight_info = {stock_list[i]: f"{best_weights[i]*100:.2f}%" for i in range(num_stocks)}

    return plot_url, weight_info

# ================= 3. 核心加载与启动逻辑 =================
def load_model():
    global global_model, global_tokenizer
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    lora_relative_path = "qwen3_4b_finance_lora"
    lora_path = os.path.join(script_dir, lora_relative_path)

    print("-" * 30)
    print("正在加载 Tokenizer...")
    global_tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side="left"
    )
    if global_tokenizer.pad_token is None:
        global_tokenizer.pad_token = global_tokenizer.eos_token

    print("正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"正在加载 LoRA 适配器: {lora_path} ...")
    if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        try:
            global_model = PeftModel.from_pretrained(model, lora_path)
            print("LoRA 加载成功！")
        except Exception as e:
            print(f"加载 LoRA 失败: {e}")
            global_model = model
    else:
        print(f"警告: 找不到 LoRA 文件夹 '{lora_path}'，将使用基础模型提供服务。")
        global_model = model

    global_model.eval()
    print("✨ 模型准备就绪！")
    print("-" * 30)

def open_browser():
    # 延迟1秒后自动打开浏览器，确保服务已启动
    webbrowser.open_new("http://127.0.0.1:5000")

def main():
    # 1. 先加载模型
    load_model()
    
    # 2. 启动新线程去打开浏览器
    threading.Timer(1.0, open_browser).start()
    
    # 3. 启动本地 Web 服务
    print("\n🚀 正在启动本地 Web 服务，请在自动弹出的浏览器中进行操作...")
    print("按 Ctrl+C 退出程序")
    app.run(host='127.0.0.1', port=5000, threaded=False)

if __name__ == "__main__":
    main()