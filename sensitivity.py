import numpy as np
import vm_placement as vp

COMBOS = [(3, 5), (5, 10), (5, 20), (10, 10), (10, 20)]   # (K, H)
SEEDS = range(10)

print(f"{'K':>3}{'H':>4}{'ovl after':>16}{'bal_std after':>18}{'migrations':>14}")
for K, H in COMBOS:
    vp.K, vp.H = K, H
    res = []
    for s in SEEDS:
        vp.rng = np.random.default_rng(s)
        L = vp.gen_trace()
        hist, mig = vp.run_predictive(L)
        m, _ = vp.metrics(L, hist, mig)
        res.append([m["after"]["overloads"], m["after"]["balance_std"], mig])
    a = np.array(res); mu, sd = a.mean(0), a.std(0)
    print(f"{K:>3}{H:>4}{mu[0]:>9.1f} ± {sd[0]:<5.1f}{mu[1]:>11.3f} ± {sd[1]:<5.3f}{mu[2]:>8.1f} ± {sd[2]:<4.1f}")