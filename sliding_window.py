import numpy as np
import vm_placement as vp
from shift_types import make_trace

def run_predictive_window(L, W=40):
    a = vp.best_fit(L[:, 0]); hist, mig = [], 0
    for t in range(vp.T):
        if t >= vp.WARM and t % vp.INTERVAL == 0:
            start = max(0, t + 1 - W)
            m = vp.train(L[:, start:t+1])
            pred = np.array([max(m.predict([L[i, t-vp.K+1:t+1]])[0], 0)
                             for i in range(vp.N_VM)])
            mig += vp.rebalance(a, pred)
        hist.append(a.copy())
    return np.array(hist), mig

POLICIES = ["pred_all", "pred_win40", "pred_win20"]

print(f"{'shift':<8}{'policy':<12}{'ovl after':>16}{'bal_std after':>18}{'migrations':>14}")
for mode in ["ramp", "jump", "burst"]:
    res = {p: [] for p in POLICIES}
    for s in range(15):
        vp.rng = np.random.default_rng(s)
        L = make_trace(mode)
        runs = {"pred_all": vp.run_predictive(L),
                "pred_win40": run_predictive_window(L, 40),
                "pred_win20": run_predictive_window(L, 20)}
        for name, (hist, mig) in runs.items():
            m, _ = vp.metrics(L, hist, mig)
            res[name].append([m["after"]["overloads"], m["after"]["balance_std"], mig])
    for name, r in res.items():
        a = np.array(r); mu, sd = a.mean(0), a.std(0)
        print(f"{mode:<8}{name:<12}{mu[0]:>9.1f} ± {sd[0]:<5.1f}"
              f"{mu[1]:>11.3f} ± {sd[1]:<5.3f}{mu[2]:>8.1f} ± {sd[2]:<4.1f}")
    print()