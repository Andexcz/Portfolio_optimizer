import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import datetime
import math


print("Stahování aktuálních tržních dat...")
tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLU']
rf_ticker = '^IRX' 

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=10*365)
all_tickers = tickers + [rf_ticker]
data = yf.download(all_tickers, start=start_date, end=end_date)['Close']

# Úprava pro různé burzy
data = data.ffill().dropna()
live_rf_rate = float(data[rf_ticker].iloc[-1]) / 100.0
print(f"Aktuální bezriziková sazba (13W T-Bill): {live_rf_rate*100:.2f} %")
prices = data[tickers]
daily_returns = prices.pct_change().dropna()


TRADING_DAYS = 252
#Používám EWMA místo EMA
mu_daily = daily_returns.ewm(halflife=252).mean().iloc[-1].values
mu = mu_daily * TRADING_DAYS
lw = LedoitWolf().fit(daily_returns)
Sigma = lw.covariance_ * TRADING_DAYS

print(f"Data úspěšně zpracována. Analyzováno {len(daily_returns)} obchodních dnů.")


def solve_portfolio(mu, Sigma, w_current=None, risk_aversion=1.0, max_weight=1.0, tc_bps=10, rf_rate=0.04, epsilon=1e-5):
    n = len(mu)
    Sigma_reg = (Sigma + Sigma.T) / 2 + epsilon * np.eye(n)
    w = cp.Variable(n)
    utility = mu @ w - risk_aversion * cp.quad_form(w, Sigma_reg)
    
    if w_current is None:
        w_current = np.zeros(n)
    
    # Transakční náklady
    tc_cost = (tc_bps / 10000) * cp.norm(w - w_current, 1)
    
    objective = cp.Maximize(utility - tc_cost)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            return None 
        
    if w.value is None:
        weights = np.ones(n) / n
    else:
       #Očištění odchylek floatu
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

# Samotná exekuce a výpis výsledků pro různé strategie
print("\n" + "=" * 70)
print("LIVE DATA MVO ANALÝZA (SEKTOROVÁ ETF)")
print("=" * 70)

print(f"Sledovaná aktiva: {', '.join(tickers)}")

# Výchozí portfolio (rovnoměrně rozložené)
current_portfolio = np.ones(len(tickers)) / len(tickers)

strategies = [
  ('Agresivní (λ=0.5)', 0.5, 1.0),
  ('Vyvážená (λ=2.0)', 2.0, 1.0),
  ('Konzervativní (λ=5.0)', 5.0, 1.0),
  ('Diverzifikovaná (25% max)', 2.0, 0.25),
]

print(f"\n{'Strategie':<26} {'Oček. výnos':<14} {'Volatilita':<12} {'Sharpe':<8} {'Obrat':<8}")
print("-" * 75)

for name, lam, max_w in strategies:
    r = solve_portfolio(
        mu, Sigma, 
        w_current=current_portfolio, 
        risk_aversion=lam, 
        max_weight=max_w,
        tc_bps=15, #oplatky (0.15 %)
        rf_rate=live_rf_rate
    )
    if r is not None:
        print(f"{name:<26} {r['return']*100:>7.2f} %       {r['volatility']*100:>6.2f} %   {r['sharpe']:>6.2f}   {r['turnover']*100:>5.1f} %")
    else:
        print(f"{name:<26} Nepodařilo se najít řešení.")

print("\nCílové váhy pro Diverzifikovanou strategii:")
div_result = solve_portfolio(mu, Sigma, w_current=current_portfolio, risk_aversion=2.0, max_weight=0.25, tc_bps=15, rf_rate=live_rf_rate)
div_weights = div_result['weights']


def vypocitej_kusy_akcii(vahy, tickery, aktualni_ceny, hodnota_portfolia):
    print(f"\n" + "=" * 68)
    print(f"Počet kusů akcií (Celkový kapitál: {hodnota_portfolia:,.2f} USD)")
    print("=" * 68)
    
    celkem_alokovano = 0
    zbyvajici_hotovost = hodnota_portfolia
    nakupy = {}
    
    print(f"{'Ticker':<8} {'Cílová váha':<12} {'Cena za kus':<13} {'Kusů ke koupi':<15} {'Alokováno (USD)':<15}")
    print("-" * 68)
    
    for ticker, vaha in zip(tickery, vahy):
            
        cilova_hodnota = hodnota_portfolia * vaha
        cena_za_kus = float(aktualni_ceny[ticker])
        
        pocet_kusu = cilova_hodnota / cena_za_kus
        skutecna_alokace = pocet_kusu * cena_za_kus
        
        nakupy[ticker] = pocet_kusu
        celkem_alokovano += skutecna_alokace
        zbyvajici_hotovost -= skutecna_alokace
        print(f"{ticker:<8} {vaha*100:>8.1f} %   {cena_za_kus:>10.2f} $   {pocet_kusu:>13.4f} ks   {skutecna_alokace:>13.2f} $")
        
    print("-" * 68)
    print(f"Zainvestováno celkem : {celkem_alokovano:>13,.2f} USD")
    print(f"Zůstatek v hotovosti : {abs(zbyvajici_hotovost):>13,.2f} USD")
    
    return nakupy

# ---------------------------------------------------------
# Spuštění
# ---------------------------------------------------------
VELIKOST_UCTU = 10000  # Velikost svého účtu v USD
nejnovejsi_ceny = data[tickers].iloc[-1]

doporuceny_nakup = vypocitej_kusy_akcii(div_weights, tickers, nejnovejsi_ceny, VELIKOST_UCTU)