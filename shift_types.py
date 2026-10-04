import numpy as np
import vm_placement as vp

def make_trace(mode):
    t = np.arange(vp.T)
    base = vp.rng.uniform(5, 20, vp.N_VM)
    L = np.array([b + 3*np.sin(2*np.pi*t/50 + vp.rng.uniform(0, 6.28))
                  + vp.rng.normal(0, 1.5, vp.T) for b in base])
    for i in vp.rng.choice(vp.N_VM, 6, replace=False):
        if mode == "ramp":
            L[i, vp.SHIFT:] += np.linspace(0, 25, vp.T - vp.SHIFT)
        elif mode == "jump":
            L[i, vp.SHIFT:] += 25
        elif mode == "burst":
            L[i, vp.SHIFT:vp.SHIFT + 15] += 25
    return np.clip(L, 1, None)

if __name__ == "__main__":
    POLICIES = ["first_fit", "best_fit", "predictive"]

    print(f"{'shift':<8}{'policy':<12}{'ovl after':>16}{'bal_std after':>18}{'migrations':>14}")
    for mode in ["ramp", "jump", "burst"]:
        res = {p: [] for p in POLICIES}
        for s in range(15):
            vp.rng = np.random.default_rng(s)
            L = make_trace(mode)
            runs = {"first_fit": vp.run_static(L, vp.first_fit),
                    "best_fit": vp.run_static(L, vp.best_fit),
                    "predictive": vp.run_predictive(L)}
            for name, (hist, mig) in runs.items():
                m, _ = vp.metrics(L, hist, mig)
                res[name].append([m["after"]["overloads"], m["after"]["balance_std"], mig])
        for name, r in res.items():
            a = np.array(r); mu, sd = a.mean(0), a.std(0)
            print(f"{mode:<8}{name:<12}{mu[0]:>9.1f} ± {sd[0]:<5.1f}"
                  f"{mu[1]:>11.3f} ± {sd[1]:<5.3f}{mu[2]:>8.1f} ± {sd[2]:<4.1f}")
        print()