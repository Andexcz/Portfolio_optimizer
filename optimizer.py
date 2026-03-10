import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import datetime
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning) # Suppress warnings for clean terminal

# ---------------------------------------------------------
# 1. Live Market Data Ingestion (Global Multi-Asset)
# ---------------------------------------------------------
print("Fetching live market data for global portfolio (Last 6 Years)...")

tickers = [
    'SPY',     # S&P 500 Proxy
    'VT',      # Vanguard Total World (FTSE All-World Proxy)
    'GLD',     # Gold Proxy
    'GOOGL',   # Alphabet
    'AMZN',    # Amazon
    'LMT',     # Lockheed Martin
    'META',    # Meta
    'ASML',    # ASML
    'NFLX',    # Netflix
    'MC.PA',   # LVMH (Paris)
    'MELI',    # MercadoLibre
    'MSFT',    # Microsoft
    'V',       # Visa
    'SIRI',    # Sirius XM
    'PYPL',    # PayPal
    'TUI1.DE', # TUI AG (Frankfurt)
    'BRK-B',   # Berkshire Hathaway Class B (Hyphen is mandatory)
    'OR.PA',   # L'Oreal (Paris)
    'SOFI',    # SoFi
    'NVO',     # Novo Nordisk
    'GRAB'     # Grab
]

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=6*365) # Changed to 6 years

# Download the Close prices
raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']

# THE QUANT FIX: Force the DataFrame columns to match our original list order
raw_data = raw_data[tickers]

# Forward-fill missing prices due to global timezone/holiday mismatches
data = raw_data.ffill().dropna()

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# ---------------------------------------------------------
# 2. Parameter Estimation (Annualized)
# ---------------------------------------------------------
TRADING_DAYS = 252

# Expected Returns & Covariance for the overall in-sample period
mu_daily = daily_returns.mean().values
mu = mu_daily * TRADING_DAYS

lw = LedoitWolf().fit(daily_returns)
Sigma = lw.covariance_ * TRADING_DAYS

print(f"Data ingested successfully. Analyzed {len(daily_returns)} trading days.")

# ---------------------------------------------------------
# 3. Production Portfolio Optimizer
# ---------------------------------------------------------
def solve_portfolio(mu, Sigma, w_current=None, risk_aversion=1.0, max_weight=1.0, tc_bps=10, rf_rate=0.04, epsilon=1e-5):
    n = len(mu)
    Sigma_reg = (Sigma + Sigma.T) / 2 + epsilon * np.eye(n)
    w = cp.Variable(n)
    
    utility = mu @ w - risk_aversion * cp.quad_form(w, Sigma_reg)
    
    if w_current is None:
        w_current = np.zeros(n)
    
    tc_cost = (tc_bps / 10000) * cp.norm(w - w_current, 1)
    
    objective = cp.Maximize(utility - tc_cost)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)
        
    if w.value is None:
        weights = np.ones(n) / n
    else:
        weights = np.clip(np.array(w.value).flatten(), 0, 1)
        weights /= weights.sum()
        
    port_return = float(np.dot(mu, weights))
    port_vol = float(np.sqrt(np.dot(weights, np.dot(Sigma, weights))))
    sharpe = (port_return - rf_rate) / port_vol if port_vol > 0 else 0
    
    return {
        'weights': weights, 
        'return': port_return, 
        'volatility': port_vol,
        'sharpe': sharpe,
        'turnover': float(np.sum(np.abs(weights - w_current)))
    }

# ---------------------------------------------------------
# 4. In-Sample Execution, Analysis & Visualization
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("IN-SAMPLE MVO DASHBOARD (6-Year Lookback)")
print("=" * 70)

live_rf_rate = 0.041 
current_portfolio = np.ones(len(tickers)) / len(tickers)

strategies = [
  ('Aggressive (λ=0.5)', 0.5, 1.0),
  ('Balanced (λ=2.0)', 2.0, 1.0),
  ('Conservative (λ=5.0)', 5.0, 1.0),
  ('Diversified (25% cap)', 2.0, 0.25),
]

results = {}

for name, lam, max_w in strategies:
    r = solve_portfolio(
        mu, Sigma, 
        w_current=current_portfolio, 
        risk_aversion=lam, 
        max_weight=max_w,
        tc_bps=15, 
        rf_rate=live_rf_rate
    )
    results[name] = r

def build_frontier(mu, Sigma, max_weight=1.0, n_pts=40):
    returns, volatilities = [], []
    n = len(mu)
    Sigma_reg = (Sigma + Sigma.T) / 2 + 1e-5 * np.eye(n)
    
    min_target = max(0.0, np.mean(mu)) 
    max_target = np.max(mu)
    
    for target in np.linspace(min_target, max_target, n_pts):
        w = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_reg)),
                          [cp.sum(w) == 1, w >= 0, w <= max_weight, mu @ w >= target])
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
                returns.append(float(mu @ w.value))
                volatilities.append(float(np.sqrt(w.value @ Sigma @ w.value)))
        except cp.error.SolverError:
            pass 
            
    return np.array(volatilities), np.array(returns)

print("Calculating Efficient Frontiers...")
vol_unc, ret_unc = build_frontier(mu, Sigma, max_weight=1.0)
vol_con, ret_con = build_frontier(mu, Sigma, max_weight=0.25)

ticker_to_category = {
    'SPY': 'Macro & Indices', 'VT': 'Macro & Indices', 'GLD': 'Macro & Indices',
    'GOOGL': 'Big Tech & Growth', 'MSFT': 'Big Tech & Growth', 'META': 'Big Tech & Growth', 
    'ASML': 'Big Tech & Growth', 'NFLX': 'Big Tech & Growth', 'CSGP': 'Big Tech & Growth',
    'AMZN': 'Consumer & Travel', 'MC.PA': 'Consumer & Travel', 'OR.PA': 'Consumer & Travel', 
    'MELI': 'Consumer & Travel', 'TUI1.DE': 'Consumer & Travel',
    'V': 'Financials & Payments', 'PYPL': 'Financials & Payments', 'SOFI': 'Financials & Payments', 'BRK-B': 'Financials & Payments',
    'LMT': 'Defensive & Other', 'NVO': 'Defensive & Other', 'SIRI': 'Defensive & Other', 'GRAB': 'Defensive & Other'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.2, 1]})

if len(vol_unc) > 0:
    ax1.plot(vol_unc * 100, ret_unc * 100, 'k--', linewidth=2, alpha=0.6, label='Frontier (Unconstrained)')
if len(vol_con) > 0:
    ax1.plot(vol_con * 100, ret_con * 100, 'gray', linestyle='-.', linewidth=2, alpha=0.8, label='Frontier (25% Cap)')

vols_annual = np.sqrt(np.diag(Sigma))
ax1.scatter(vols_annual * 100, mu * 100, color='gray', s=50, alpha=0.4, label='Individual Assets')

key_tickers = ['SPY', 'GLD', 'MSFT', 'NVO', 'META', 'BRK-B', 'PYPL']
for i, ticker in enumerate(tickers):
    if ticker in key_tickers:
        ax1.annotate(ticker, (vols_annual[i] * 100 + 0.5, mu[i] * 100), fontsize=9, alpha=0.8, fontweight='bold')

colors = ['red', 'blue', 'green', 'purple']
markers = ['o', 's', '^', 'D']
for (name, r), color, marker in zip(results.items(), colors, markers):
    ax1.scatter(r['volatility'] * 100, r['return'] * 100, color=color, s=200, marker=marker, 
                edgecolors='black', linewidth=1.5, label=name, zorder=10)

ax1.set_title('Efficient Frontier & Strategy Portfolios (6 Years)', fontweight='bold', fontsize=14)
ax1.set_xlabel('Annualized Volatility (%)', fontsize=11)
ax1.set_ylabel('Annualized Expected Return (%)', fontsize=11)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

names = list(results.keys())
weights_matrix = np.array([results[name]['weights'] for name in names])

unique_categories = list(dict.fromkeys(ticker_to_category.values())) 
grouped_weights = {cat: np.zeros(len(names)) for cat in unique_categories}

for i, ticker in enumerate(tickers):
    cat = ticker_to_category[ticker]
    grouped_weights[cat] += weights_matrix[:, i] * 100

bottom = np.zeros(len(names))
cat_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_categories)))

for i, cat in enumerate(unique_categories):
    ax2.bar(names, grouped_weights[cat], bottom=bottom, color=cat_colors[i], label=cat, edgecolor='white', width=0.6)
    bottom += grouped_weights[cat]

ax2.set_title('Capital Allocation by Macro Category', fontweight='bold', fontsize=14)
ax2.set_ylabel('Weight (%)', fontsize=11)
ax2.tick_params(axis='x', rotation=15)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.show(block=False) # Plot without stopping the backtester
plt.pause(2)

# ---------------------------------------------------------
# 5. Out-of-Sample Walk-Forward Backtest (MVO vs. Risk Parity)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("INITIALIZING WALK-FORWARD BACKTEST (6-Year Data)")
print("=" * 70)

LOOKBACK_DAYS = 252      # 1 year of training data
REBALANCE_DAYS = 21      # 1 month holding period
TC_BPS = 15              
RF_RATE = 0.041          

n_days, n_assets = daily_returns.shape
dates = daily_returns.index

bt_returns_div = [] 
bt_returns_rp = []  
bt_returns_ew = []  
bt_dates = []

w_current_div = np.ones(n_assets) / n_assets
w_current_rp = np.ones(n_assets) / n_assets
w_ew = np.ones(n_assets) / n_assets

print(f"Running 5-Year Out-of-Sample Backtest from {dates[LOOKBACK_DAYS].date()} to {dates[-1].date()}...")
print("Rebalancing every 21 days. Let the battle begin...")

for t in range(LOOKBACK_DAYS, n_days, REBALANCE_DAYS):
    train_data = daily_returns.iloc[t - LOOKBACK_DAYS : t]
    end_t = min(t + REBALANCE_DAYS, n_days)
    test_data = daily_returns.iloc[t : end_t]
    
    mu_train = train_data.mean().values * 252
    
    try:
        lw_train = LedoitWolf().fit(train_data)
        Sigma_train = lw_train.covariance_ * 252
    except:
        Sigma_train = train_data.cov().values * 252 
        
    # --- STRATEGY 1: Diversified MVO ---
    opt_res = solve_portfolio(
        mu_train, Sigma_train, 
        w_current=w_current_div, 
        risk_aversion=2.0, max_weight=0.25, tc_bps=TC_BPS, rf_rate=RF_RATE
    )
    w_target_div = opt_res['weights']
    
    # --- STRATEGY 2: THE OPPONENT (Inverse Volatility / Risk Parity) ---
    volatilities = np.sqrt(np.diag(Sigma_train))
    inv_vol = 1.0 / volatilities
    w_target_rp = inv_vol / np.sum(inv_vol) 
    
    tc_div = (TC_BPS / 10000) * np.sum(np.abs(w_target_div - w_current_div))
    tc_rp = (TC_BPS / 10000) * np.sum(np.abs(w_target_rp - w_current_rp))
    
    period_rets_div = test_data.dot(w_target_div).values
    period_rets_rp = test_data.dot(w_target_rp).values
    period_rets_ew = test_data.dot(w_ew).values
    
    if len(period_rets_div) > 0:
        period_rets_div[0] -= tc_div
        period_rets_rp[0] -= tc_rp
        tc_ew = (TC_BPS / 10000) * np.sum(np.abs(w_ew - w_current_div)) 
        period_rets_ew[0] -= tc_ew
        
    bt_returns_div.extend(period_rets_div)
    bt_returns_rp.extend(period_rets_rp)
    bt_returns_ew.extend(period_rets_ew)
    bt_dates.extend(test_data.index)
    
    w_current_div = w_target_div
    w_current_rp = w_target_rp

print("Backtest complete! Generating performance metrics...")

strat_returns = pd.Series(bt_returns_div, index=bt_dates)
rp_returns = pd.Series(bt_returns_rp, index=bt_dates)
ew_returns = pd.Series(bt_returns_ew, index=bt_dates)

spy_idx = tickers.index('SPY')
spy_returns = daily_returns.iloc[LOOKBACK_DAYS:, spy_idx]

eq_strat = (1 + strat_returns).cumprod()
eq_rp = (1 + rp_returns).cumprod()
eq_ew = (1 + ew_returns).cumprod()
eq_spy = (1 + spy_returns).cumprod()

def calc_mdd(cum_returns):
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1.0
    return drawdown.min()

sharpe_strat = np.sqrt(252) * (strat_returns.mean() / strat_returns.std())
sharpe_rp = np.sqrt(252) * (rp_returns.mean() / rp_returns.std())
sharpe_ew = np.sqrt(252) * (ew_returns.mean() / ew_returns.std())
sharpe_spy = np.sqrt(252) * (spy_returns.mean() / spy_returns.std())

print("\n" + "-" * 75)
print("OUT-OF-SAMPLE PERFORMANCE (Post-Fees)")
print("-" * 75)
print(f"{'Metric':<18} | {'Div MVO':<12} | {'Risk Parity':<12} | {'Equal Wt':<10} | {'SPY':<10}")
print("-" * 75)
print(f"{'Total Return':<18} | {(eq_strat.iloc[-1]-1)*100:>11.1f}% | {(eq_rp.iloc[-1]-1)*100:>11.1f}% | {(eq_ew.iloc[-1]-1)*100:>9.1f}% | {(eq_spy.iloc[-1]-1)*100:>9.1f}%")
print(f"{'Max Drawdown':<18} | {calc_mdd(eq_strat)*100:>11.1f}% | {calc_mdd(eq_rp)*100:>11.1f}% | {calc_mdd(eq_ew)*100:>9.1f}% | {calc_mdd(eq_spy)*100:>9.1f}%")
print(f"{'OOS Sharpe Ratio':<18} | {sharpe_strat:>11.2f}  | {sharpe_rp:>11.2f}  | {sharpe_ew:>9.2f}  | {sharpe_spy:>9.2f}")

plt.figure(figsize=(14, 7))
plt.plot(eq_strat.index, eq_strat, label=f'Diversified MVO (25% Cap)', color='purple', linewidth=2.5)
plt.plot(eq_rp.index, eq_rp, label='The Opponent: Risk Parity', color='orange', linewidth=2.5)
plt.plot(eq_ew.index, eq_ew, label='Equal Weight Benchmark', color='gray', linestyle='--', alpha=0.7)
plt.plot(eq_spy.index, eq_spy, label='SPY (S&P 500)', color='blue', linestyle=':', alpha=0.7)

plt.title('Out-of-Sample Walk-Forward Backtest (MVO vs. Risk Parity)', fontweight='bold', fontsize=15)
plt.ylabel('Cumulative Growth ($1 Invested)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Current Target Allocations (For Execution)
# ---------------------------------------------------------
print("\n" + "=" * 55)
print("CURRENT TARGET ALLOCATIONS (Execute Today)")
print("=" * 55)

# Extract the latest MVO weights from our In-Sample results (Section 4)
latest_mvo_weights = results['Diversified (25% cap)']['weights']

# Calculate the latest Risk Parity weights using the full 6-year covariance
latest_volatilities = np.sqrt(np.diag(Sigma))
latest_inv_vol = 1.0 / latest_volatilities
latest_rp_weights = latest_inv_vol / np.sum(latest_inv_vol)

print(f"{'Ticker':<10} | {'Macro Category':<22} | {'Div MVO':<10} | {'Risk Parity':<12}")
print("-" * 65)

for i, ticker in enumerate(tickers):
    # Only print if the weight is meaningful (greater than 0.05%) to avoid clutter
    mvo_w = latest_mvo_weights[i] * 100
    rp_w = latest_rp_weights[i] * 100
    cat = ticker_to_category[ticker]
    
    # Format with a clean zero for empty MVO slots
    mvo_str = f"{mvo_w:>8.1f}%" if mvo_w > 0.1 else f"{'0.0%':>9}"
    
    print(f"{ticker:<10} | {cat:<22} | {mvo_str} | {rp_w:>9.1f}%")

print("-" * 65)
print(f"{'TOTAL':<10} | {'':<22} | {np.sum(latest_mvo_weights)*100:>8.0f}% | {np.sum(latest_rp_weights)*100:>9.0f}%")