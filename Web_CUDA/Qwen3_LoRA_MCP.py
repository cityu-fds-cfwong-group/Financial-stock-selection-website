import os
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

load_dotenv("API.env")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
print("DEBUG: Loaded API key =", os.getenv("ALPHA_VANTAGE_API_KEY"))
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("请在 API.env 文件中设置 ALPHA_VANTAGE_API_KEY")

model_name = "Qwen/Qwen3-4B-Thinking-2507"
lora_path = "./qwen3_4b_finance_lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

def fetch_stock_overview(symbol: str):
    """从 Alpha Vantage 获取公司概览（Global Quote + Company Overview）"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        quote = data.get("Global Quote", {})
        if quote:
            price = quote.get("05. price")
            change = quote.get("09. change")
            change_percent = quote.get("10. change percent")
            return f"股票 {symbol}: 当前价格 ${price}, 涨跌 {change} ({change_percent})"
        else:
            url2 = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            resp2 = requests.get(url2, timeout=10)
            overview = resp2.json()
            name = overview.get("Name", "Unknown")
            market_cap = overview.get("MarketCapitalization", "N/A")
            sector = overview.get("Sector", "N/A")
            return f"{symbol} ({name}): 市值 {market_cap}, 所属板块 {sector}"
    except Exception as e:
        return f"无法获取 {symbol} 的数据: {str(e)}"

user_input = "比较今天AAPL和NVDA的市值和股票价格"

tickers_to_fetch = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"] # 需预设一些公司的股票代码

financial_context = "\n".join([fetch_stock_overview(ticker) for ticker in tickers_to_fetch])

enhanced_prompt = (
    f"以下是当前部分大型科技公司的最新市场数据：\n{financial_context}\n\n"
    f"基于以上信息和你的知识，请回答以下问题：\n{user_input}"
)

messages = [{"role": "user", "content": enhanced_prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

output_file = "qwen3_response.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"提问：\n{user_input}\n\n")
    f.write(f"附加金融数据：\n{financial_context}\n\n")
    f.write(f"模型回答：\n{response}\n")

print(f"模型回答已保存到：{os.path.abspath(output_file)}")