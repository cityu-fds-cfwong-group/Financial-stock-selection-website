# -*- coding: utf-8 -*-
"""
马科维兹有效前沿分析（A股10只股票5天数据版）
数据来源：A股数据_2026_3_11 (1).csv（10只股票，2026-03-05至2026-03-11共5个交易日）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import os
warnings.filterwarnings('ignore')

# ======================== 1. 中文显示配置 ========================
def setup_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
    font_list = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            break
        except:
            continue

setup_chinese_font()

# ======================== 2. 读取A股数据并重构（核心适配） ========================
def get_a_stock_data():
    """
    读取A股10只股票5天数据，重构为马科维兹分析格式：
    行：日期，列：股票名称，值：最新价（收盘价）
    """
    # 你的文件路径（桌面的A股数据文件，注意括号转义/直接写文件名）
    file_path = r"C:/Users/LSY/Desktop/web_3/A股数据_2026_3_11 (1).csv"
    # 若文件名识别失败，可尝试直接用文件名（文件在桌面时）
    # file_path = "A股数据_2026_3_11 (1).csv"

    try:
        # 1. 验证文件是否存在
        if not os.path.exists(file_path):
            # 尝试桌面默认路径兜底
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            file_path = os.path.join(desktop, "A股数据_2026_3_11 (1).csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"未在桌面找到文件：A股数据_2026_3_11 (1).csv")
        
        # 2. 读取CSV文件
        print(f"正在读取A股数据文件：{file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ 文件读取成功，原始数据形状：{df.shape}")
        
        # 3. 数据清洗：仅保留需要的列（日期、名称、最新价）
        df_clean = df[['日期', '名称', '最新价']].copy()
        # 转换日期格式，去重排序
        df_clean['日期'] = pd.to_datetime(df_clean['日期'])
        df_clean = df_clean.drop_duplicates(subset=['日期', '名称']).sort_values(['日期', '名称'])
        
        # 4. 重构数据：透视表（行=日期，列=股票名称，值=最新价）
        df_pivot = df_clean.pivot(index='日期', columns='名称', values='最新价')
        print(f"✅ 数据重构完成，分析用数据形状：{df_pivot.shape}")
        print(f"📅 交易日期：{[d.strftime('%Y-%m-%d') for d in df_pivot.index]}")
        print(f"📈 股票列表：{list(df_pivot.columns)}")
        
        # 5. 计算日收益率（简单收益率，适配短数据）
        returns_df = df_pivot.pct_change().dropna()
        print(f"📊 收益率数据生成完成：{len(returns_df)}个有效交易周期，{len(returns_df.columns)}只股票")
        
        return returns_df

    except Exception as e:
        print(f"❌ A股数据读取失败：{type(e).__name__} - {str(e)}")
        print("👉 自动切换到模拟数据应急")
        return generate_sim_data()

def generate_sim_data():
    """备用：模拟10只股票的收益率数据（适配5天短周期）"""
    np.random.seed(42)
    n_assets = 10  # 10只股票
    n_days = 4     # 5天价格对应4个收益率
    
    # 模拟A股短期收益率特征（贴合你数据的涨跌幅度）
    mean_returns = np.random.uniform(low=-0.05, high=0.08, size=n_assets) / n_days
    volatilities = np.random.uniform(low=0.03, high=0.12, size=n_assets) / np.sqrt(n_days)
    
    # 生成协方差矩阵（保证半正定，避免数值错误）
    corr_matrix = np.random.rand(n_assets, n_assets)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals[eigenvals < 0] = 0
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # 生成收益率数据
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns_data, columns=[f'股票{i+1}' for i in range(n_assets)])
    print(f"✅ 模拟数据生成完成：{len(returns_df)}个交易周期，{len(returns_df.columns)}只股票")
    return returns_df

# ======================== 3. 马科维兹核心计算（原逻辑保留，适配短数据） ========================
def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    """计算组合收益率、波动率、夏普比率（无风险利率=0，适配短数据年化）"""
    n_periods = len(mean_returns) if isinstance(mean_returns, pd.Series) else 4  # 固定4个收益率周期
    annual_factor = 252 / n_periods  # 按实际周期折算年化（5天数据≈年化63倍）
    port_return = np.sum(mean_returns * weights) * annual_factor  # 年化收益率
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annual_factor)  # 年化波动率
    sharpe_ratio = port_return / port_vol if port_vol != 0 else 0  # 夏普比率（无风险利率=0）
    return port_return, port_vol, sharpe_ratio

def generate_random_portfolios(n_portfolios, mean_returns, cov_matrix):
    """生成10000个随机投资组合（原逻辑保留）"""
    results = np.zeros((3, n_portfolios))
    weights_record = []
    n_assets = len(mean_returns)
    
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))  # 权重和为1，非负
        weights_record.append(weights)
        port_return, port_vol, sharpe_ratio = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = sharpe_ratio
    
    print(f"🎲 随机组合生成完成：{n_portfolios}个组合")
    return results, weights_record

def minimize_volatility(target_return, mean_returns, cov_matrix):
    """给定收益率，求解最小波动率组合（原逻辑保留）"""
    n_assets = len(mean_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: calculate_portfolio_metrics(x, mean_returns, cov_matrix)[0] - target_return}
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # 不允许卖空
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        lambda x: calculate_portfolio_metrics(x, mean_returns, cov_matrix)[1],
        initial_weights, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'disp': False, 'ftol': 1e-6}  # 放宽收敛条件，适配短数据
    )
    return result

def get_efficient_frontier(mean_returns, cov_matrix, n_points=30):
    """生成有效前沿曲线（适配短数据，减少点数）"""
    min_return = min(mean_returns) * (252/4)
    max_return = max(mean_returns) * (252/4)
    target_returns = np.linspace(min_return, max_return, n_points)
    
    efficient_vols = []
    efficient_returns = []
    for target in target_returns:
        result = minimize_volatility(target, mean_returns, cov_matrix)
        if result.success:
            port_return, port_vol, _ = calculate_portfolio_metrics(result.x, mean_returns, cov_matrix)
            efficient_returns.append(port_return)
            efficient_vols.append(port_vol)
    
    # 清理无效点
    efficient_returns = np.array(efficient_returns)
    efficient_vols = np.array(efficient_vols)
    valid_mask = ~np.isnan(efficient_vols)
    efficient_returns = efficient_returns[valid_mask]
    efficient_vols = efficient_vols[valid_mask]
    
    print(f"📊 有效前沿生成完成：{len(efficient_returns)}个有效组合点")
    return efficient_returns, efficient_vols

def maximize_sharpe_ratio(mean_returns, cov_matrix):
    """求解夏普比率最大化的最优组合（原逻辑保留）"""
    n_assets = len(mean_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        lambda x: -calculate_portfolio_metrics(x, mean_returns, cov_matrix)[2],
        initial_weights, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'disp': False, 'ftol': 1e-6}
    )
    return result

# ======================== 4. 主程序执行（适配A股10只股票数据） ========================
if __name__ == "__main__":
    # 1. 获取A股数据（核心替换：读取10只股票5天数据）
    print("="*60)
    print("第一步：读取A股10只股票5天交易数据")
    print("="*60)
    returns_df = get_a_stock_data()
    mean_returns = returns_df.mean().values  # 资产平均收益率
    cov_matrix = returns_df.cov().values     # 收益率协方差矩阵
    n_assets = len(returns_df.columns)       # 股票数量（10只）
    print(f"\n📋 数据基础信息：")
    print(f"   - 股票数量：{n_assets}只")
    print(f"   - 有效收益周期：{len(returns_df)}个（5天价格对应）")
    print(f"   - 股票名称：{list(returns_df.columns)}")
    
    # 2. 核心计算（随机组合+有效前沿+最优组合）
    print("\n" + "="*60)
    print("第二步：马科维兹组合优化计算")
    print("="*60)
    n_portfolios = 10000  # 随机组合数量
    random_results, weights_record = generate_random_portfolios(n_portfolios, mean_returns, cov_matrix)
    efficient_returns, efficient_vols = get_efficient_frontier(mean_returns, cov_matrix)
    
    # 最优组合（最大化夏普比率）
    optimal_result = maximize_sharpe_ratio(mean_returns, cov_matrix)
    optimal_weights = optimal_result.x
    optimal_return, optimal_vol, optimal_sharpe = calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
    
    # 最小方差组合
    min_vol_result = minimize(
        lambda x: calculate_portfolio_metrics(x, mean_returns, cov_matrix)[1],
        np.ones(n_assets)/n_assets,
        method='SLSQP', bounds=tuple((0,1) for _ in range(n_assets)),
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x)-1}],
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    min_vol_return, min_vol_vol, _ = calculate_portfolio_metrics(min_vol_result.x, mean_returns, cov_matrix)
    
    print(f"\n✅ 核心计算完成：")
    print(f"   - 最优组合夏普比率：{optimal_sharpe:.2f}")
    print(f"   - 最小方差组合年化波动率：{min_vol_vol:.2%}")
    
    # 3. 可视化（适配A股数据，图表保存至桌面）
    print("\n" + "="*60)
    print("第三步：绘制马科维兹有效前沿图表")
    print("="*60)
    plt.figure(figsize=(14, 9))
    
    # 绘制随机组合散点图（颜色=夏普比率）
    scatter = plt.scatter(
        random_results[1, :], random_results[0, :],
        c=random_results[2, :], cmap='viridis', alpha=0.6, s=30
    )
    
    # 绘制有效前沿曲线
    plt.plot(efficient_vols, efficient_returns, 'r-', linewidth=3, label='有效前沿')
    
    # 标记关键组合
    plt.scatter(optimal_vol, optimal_return, color='red', s=250, marker='*', zorder=10,
                label=f'最优组合（夏普率：{optimal_sharpe:.2f}）')
    plt.scatter(min_vol_vol, min_vol_return, color='darkblue', s=200, marker='o', zorder=10,
                label='最小方差组合')
    
    # 图表美化
    cbar = plt.colorbar(scatter)
    cbar.set_label('夏普比率', fontsize=14)
    plt.xlabel('年化波动率（风险）', fontsize=16, fontweight='bold')
    plt.ylabel('年化收益率', fontsize=16, fontweight='bold')
    plt.title('马科维兹有效前沿曲线（A股10只股票5天数据）', fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    
    # 标注最优组合权重（适配10只股票，自动调整位置）
    weight_text = "最优组合权重：\n" + "\n".join([
        f"{returns_df.columns[i]}: {optimal_weights[i]:.2%}" for i in range(n_assets) if optimal_weights[i] > 0.001
    ])
    # 权重标注位置自适应，避免超出图表
    text_x = optimal_vol + (np.max(random_results[1, :])-np.min(random_results[1, :]))*0.05
    text_y = optimal_return - (np.max(random_results[0, :])-np.min(random_results[0, :]))*0.1
    plt.annotate(
        weight_text, xy=(optimal_vol, optimal_return),
        xytext=(text_x, text_y), fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='black'),
        arrowprops=dict(arrowstyle='->', color='red', linewidth=2, alpha=0.8), zorder=15
    )
    
    # 保存图表到桌面
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop, "A股10只股票_马科维兹有效前沿.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"🖼️  有效前沿图表已保存至桌面：{save_path}")
    
    # 4. 输出详细结果（A股数据专属）
    print("\n" + "="*80)
    print("马科维兹组合优化结果（A股10只股票5天交易数据）")
    print("="*80)
    print(f"📅 交易时间：2026-03-05 ~ 2026-03-11（5个交易日，4个收益周期）")
    print(f"📈 分析标的：{n_assets}只A股股票")
    print(f"⚖️  约束条件：不允许卖空（权重0~1），权重总和=1")
    print(f"📊 年化折算：按252个交易日折算（短数据年化结果仅作相对参考）\n")
    
    print(f"【最优组合（最大化夏普比率）】")
    print(f"   年化收益率：{optimal_return:.2%}")
    print(f"   年化波动率：{optimal_vol:.2%}")
    print(f"   夏普比率：{optimal_sharpe:.2f}\n")
    
    print(f"【最小方差组合（最小风险）】")
    print(f"   年化收益率：{min_vol_return:.2%}")
    print(f"   年化波动率：{min_vol_vol:.2%}\n")
    
    print(f"【最优组合权重分配（按权重降序）】")
    print("-"*60)
    weight_df = pd.DataFrame({
        '股票名称': returns_df.columns,
        '权重占比': [f"{w:.2%}" for w in optimal_weights],
        '权重数值': optimal_weights.round(4)
    }).sort_values('权重数值', ascending=False)
    print(weight_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("⚠️  重要提示：本结果基于5天短期数据，年化值仅作**相对对比参考**，不代表长期实际收益！")
    print("⚠️  实际投资分析建议使用2~5年日度数据，结果更具参考价值。")
    print("="*80)