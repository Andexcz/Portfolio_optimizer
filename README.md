# Kvantitativní alokace portfolia: Markowitzova optimalizace v praxi

Tento projekt je zaměřený na dynamickou alokaci kapitálu a portfolio management. Jde o implementaci klasické Mean-Variance optimalizace (MVO) aplikované na sektorová ETF (XLK, XLF, XLE, XLV, XLP, XLU). 


## 📂 Co tu najdete

* `optimizer.py` - Alokační kalkulátor (Snapshot). Nástroj, který si stáhne aktuální data a vypočítá přesné váhy a počet kusů ETF pro nákup.
* `MVO_backtester.py` - backtester na 10leté historii s pravidelným měsíčním rebalancováním.

---

## 🛠 (`optimizer.py`)

projekt `optimizer.py` slouží jako exekuční nástroj pro aktuální den. Řeší maximalizaci profitu portfolia pomocí konvexní optimalizace (využívám knihovnu `cvxpy`).

**Klíčové vlastnosti:**
* **Ledoit-Wolf Shrinkage:** Historická kovarianční matice obsahuje obrovské množství šumu. K její stabilizaci používám odhad podle Ledoita a Wolfa.
* **Odhady výnosů:** Místo EMA používám EWMA (Exponenciálně weighted mean average), abych dal větší váhu nedávné historii a lépe zachytil aktuální tržní momentum.
* **Transakční náklady:** Poplatky jsou přímo integrovány do optimalizační funkce pomocí $L_1$ normy (`cp.norm(w - w_current, 1)`). Model je tak matematicky penalizován za zbytečné přelévání kapitálu (turnover) a rebalancuje, jen když to prokazatelně vede k leopšímu výsledku.

---

## 📊 Backtest a realita diverzifikace (`MVO_backtester.py`)

Skript `MVO_backtester.py` bere matematiku z optimizeru a posouvá ji měsíc po měsíci o 10 let zpět. Použil jsem striktní averzi k riziku ($\lambda = 2.0$) a diverzifikační limit: **max 25 % kapitálu do jednoho sektoru**.

**Horší výsledky než S&P 500 porazilo:**
Při pohledu na výsledky backtestu je vidět, že za posledních 10 let tento model skončil v profitu cca +182 %, zatímco prosté držení S&P 500 udělalo přes +200 %. 

To je sice hezké, protože jsem ani nečekal, že základní kód porazí trh. Je to však dokonalá ukázka toho, jak fungují risk limity a diverzifikace v praxi, což jsem se chtěl vlastně naučit. Index S&P 500 se v posledních letech stal de facto "technologickým" (bráno s rezervou) fondem s obrovskou koncentrací v pár největších firmách. Můj model dodržoval 25% limit na sektor. Když technologie prudce rostly, můj model nemohl přelít všechen kapitál do technologických firem, ale musel alokovat kapitál i do defenzivnějších, pomaleji rostoucích sektorů.

**Zajímavé zjištění (Krize 2022):**
Síla modelu se ukazuje hlavně v krizových obdobích, jako byl rok 2022. Zatímco index SP500 zaznamenal výrazný propad, MVO model včas identifikoval změnu volatility a rotoval kapitál do energetiky (XLE) a utilit (XLU). Tím propad portfolia výrazně vyhladil a ochránil kapitál.

Kdybych limit 25 % odstranil a snížil averzi k riziku, SPY pravděpodobně hrubou silou překonám, ale zničím tím původní myšlenku stabilního a bezpečně diverzifikovaného portfolia s nižším drawdownem, než mělo SP500.