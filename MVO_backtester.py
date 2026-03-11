import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import datetime

class MVO_WalkForwardBacktester:
    def __init__(self, tickers, benchmark='SPY', lookback_days=252, rebalance_freq='M', tc_bps=15, risk_aversion=2.0, max_weight=0.25):
        self.tickers = tickers
        self.benchmark = benchmark
        self.lookback = lookback_days
        self.tc_bps = tc_bps
        self.lam = risk_aversion
        self.max_weight = max_weight
        
        self.prices = None
        self.returns = None
        self.results = None
        
    def fetch_data(self, years=10):
        print(f"Stahuji data pro {len(self.tickers)} tickerů a benchmark {self.benchmark}...")
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=years*365)
        
        all_tickers = self.tickers + [self.benchmark]
        data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
        data = data.ffill().dropna()
        
        self.prices = data
        self.returns = data.pct_change().dropna()
        print("Data úspěšně stažena a vyčištěna.")

    def optimize_portfolio(self, mu, Sigma, w_current):
        n = len(mu)
        Sigma_reg = (Sigma + Sigma.T) / 2 + 1e-5 * np.eye(n)
        w = cp.Variable(n)
        
        utility = mu @ w - self.lam * cp.quad_form(w, Sigma_reg)
        tc_cost = (self.tc_bps / 10000) * cp.norm(w - w_current, 1)
        
        objective = cp.Maximize(utility - tc_cost)
        constraints = [cp.sum(w) == 1, w >= 0, w <= self.max_weight]
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            try:
                prob.solve(solver=cp.SCS)
            except:
                return w_current # Pokud selže solver, držíme původní váhy
            
        if w.value is None:
            return w_current
            
        weights = np.clip(np.array(w.value).flatten(), 0, 1)
        return weights / weights.sum()

    def run_backtest(self):
        print("Spouštím Walk-Forward Backtest (Měsíční rebalancování)...")
        
        # Získání dat pouze pro naše ETF (bez benchmarku)
        asset_returns = self.returns[self.tickers]
        
        # Určení dnů pro rebalancování (konec každého měsíce)
        rebalance_dates = asset_returns.resample('ME').last().index
        
        w_current = np.ones(len(self.tickers)) / len(self.tickers)
        portfolio_value = 10000.0
        
        equity_curve = []
        dates = []
        
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i+1]
            
            # Lookback window pro trénink
            train_data = asset_returns.loc[:current_date].tail(self.lookback)
            
            if len(train_data) < self.lookback:
                continue # Přeskočíme, dokud nemáme dost historie
                
            # Výpočet mu a Sigma
            mu_daily = train_data.ewm(halflife=252).mean().iloc[-1].values
            mu = mu_daily * 252
            
            lw = LedoitWolf().fit(train_data)
            Sigma = lw.covariance_ * 252
            
            # Optimalizace
            w_new = self.optimize_portfolio(mu, Sigma, w_current)
            
            # Stržení poplatků za změnu vah
            turnover = np.sum(np.abs(w_new - w_current))
            tc_cash = portfolio_value * turnover * (self.tc_bps / 10000)
            portfolio_value -= tc_cash
            w_current = w_new
            
            # Simulace držení po další měsíc
            period_returns = asset_returns.loc[current_date:next_date].iloc[1:]
            for date, daily_ret in period_returns.iterrows():
                # Výnos portfolia v daný den
                port_ret = np.dot(w_current, daily_ret.values)
                portfolio_value *= (1 + port_ret)
                
                # Aktualizace vah vlivem tržních pohybů (drift)
                w_current = w_current * (1 + daily_ret.values)
                w_current /= w_current.sum()
                
                equity_curve.append(portfolio_value)
                dates.append(date)

        # Výpočet benchmarku - přesné napárování na naše obchodní dny
        bench_ret = self.returns[self.benchmark].loc[dates]
        bench_equity = 10000.0 * (1 + bench_ret).cumprod()

        self.results = pd.DataFrame({
            'MVO_Portfolio': equity_curve,
            'Benchmark_SPY': bench_equity.values
        }, index=dates)
        
        print("Backtest dokončen!")

    def plot_results(self):
        if self.results is None:
            return
            
        mvo_ret = self.results['MVO_Portfolio'].iloc[-1] / 10000.0 - 1
        spy_ret = self.results['Benchmark_SPY'].iloc[-1] / 10000.0 - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['MVO_Portfolio'], label=f'MVO Portfolio (Výnos: {mvo_ret*100:.2f} %)', color='blue')
        plt.plot(self.results.index, self.results['Benchmark_SPY'], label=f'S&P 500 Benchmark (Výnos: {spy_ret*100:.2f} %)', color='gray', alpha=0.7)
        
        plt.title('Walk-Forward Backtest: Markowitzova optimalizace vs S&P 500')
        plt.ylabel('Hodnota účtu (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Tickers: Tech, Finance, Energetika, Zdravotnictví, Spotřební zboží, Utility
    etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLU']
    
    backtester = MVO_WalkForwardBacktester(
        tickers=etfs, 
        risk_aversion=2.0, 
        max_weight=0.25, # Max 25 % kapitálu do jednoho sektoru
        tc_bps=15        # Transakční poplatky 0.15 %
    )
    
    backtester.fetch_data(years=10)
    backtester.run_backtest()
    backtester.plot_results()