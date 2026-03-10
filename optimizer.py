import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import datetime

# ---------------------------------------------------------
# 1. Live Market Data Ingestion
# ---------------------------------------------------------
print("Fetching live market data...")

# SPDR Sector ETFs mapped to our previous categories
tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLU']
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5*365) # 5 years of history

# Download Adjusted Close prices
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns and drop missing values
daily_returns = data.pct_change().dropna()

# ---------------------------------------------------------
# 2. Parameter Estimation (Annualized)
# ---------------------------------------------------------
# Quants annualize daily returns and covariance for standard MVO
TRADING_DAYS = 252

# Expected Returns (Historical Mean)
mu_daily = daily_returns.mean().values
mu = mu_daily * TRADING_DAYS

# Robust Covariance (Ledoit-Wolf Shrinkage)
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
    
    # Transaction cost penalty
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
# 4. Execution & Analysis
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("LIVE DATA MVO ANALYSIS (SECTOR ETFs)")
print("=" * 70)

# Print current live assets
print(f"Assets: {', '.join(tickers)}")

# Current Risk-Free Rate (Approximate current 10-Year Treasury Yield)
live_rf_rate = 0.041 

# Assume an equal-weight starting portfolio
current_portfolio = np.ones(len(tickers)) / len(tickers)

strategies = [
  ('Aggressive (λ=0.5)', 0.5, 1.0),
  ('Balanced (λ=2.0)', 2.0, 1.0),
  ('Conservative (λ=5.0)', 5.0, 1.0),
  ('Diversified (25% cap)', 2.0, 0.25),
]

print(f"\n{'Strategy':<22} {'Expected Ret':<14} {'Vol':<9} {'Sharpe':<8} {'Turnover':<8}")
print("-" * 70)
for name, lam, max_w in strategies:
    r = solve_portfolio(
        mu, Sigma, 
        w_current=current_portfolio, 
        risk_aversion=lam, 
        max_weight=max_w,
        tc_bps=15, 
        rf_rate=live_rf_rate
    )
    print(f"{name:<22} {r['return']*100:>7.2f}%       {r['volatility']*100:>5.2f}%   {r['sharpe']:>6.2f}   {r['turnover']*100:>5.1f}%")

print("\nOptimized Weights for Diversified Strategy:")
div_weights = solve_portfolio(mu, Sigma, w_current=current_portfolio, risk_aversion=2.0, max_weight=0.25, tc_bps=15, rf_rate=live_rf_rate)['weights']
for ticker, weight in zip(tickers, div_weights):
    print(f"{ticker}: {weight*100:>5.1f}%")