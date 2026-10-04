import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
T, N_VM, N_HOST, CAP = 200, 20, 6, 100
SHIFT, K, H, INTERVAL, WARM = 100, 5, 10, 10, 30

def gen_trace():
    t = np.arange(T)
    base = rng.uniform(5, 20, N_VM)
    L = np.array([b + 3*np.sin(2*np.pi*t/50 + rng.uniform(0, 6.28))
                  + rng.normal(0, 1.5, T) for b in base])
    for i in rng.choice(N_VM, 6, replace=False):      # demand shift
        L[i, SHIFT:] += np.linspace(0, 25, T - SHIFT)
    return np.clip(L, 1, None)

def first_fit(d):
    free = np.full(N_HOST, float(CAP)); a = np.zeros(N_VM, int)
    for i, x in enumerate(d):
        h = next((h for h in range(N_HOST) if free[h] >= x), int(free.argmax()))
        a[i] = h; free[h] -= x
    return a

def best_fit(d):
    free = np.full(N_HOST, float(CAP)); a = np.zeros(N_VM, int)
    for i, x in enumerate(d):
        c = [h for h in range(N_HOST) if free[h] >= x]
        h = min(c, key=lambda h: free[h] - x) if c else int(free.argmax())
        a[i] = h; free[h] -= x
    return a

def run_static(L, fn):
    a = fn(L[:, 0]); return np.array([a] * T), 0

def train(hist):
    X, y = [], []
    for i in range(N_VM):
        for s in range(hist.shape[1] - K - H + 1):
            X.append(hist[i, s:s+K]); y.append(hist[i, s+K+H-1])
    return LinearRegression().fit(X, y)

def rebalance(a, pred, thr=0.85):
    moved = 0
    for h in range(N_HOST):
        while pred[a == h].sum() > thr * CAP:
            vms = np.where(a == h)[0]
            if len(vms) <= 1: break
            v = vms[np.argmax(pred[vms])]
            tgt = min(range(N_HOST), key=lambda x: pred[a == x].sum())
            if tgt == h or pred[a == tgt].sum() + pred[v] > thr * CAP: break
            a[v] = tgt; moved += 1
    return moved

def run_predictive(L):
    a = best_fit(L[:, 0]); hist, mig = [], 0
    for t in range(T):
        if t >= WARM and t % INTERVAL == 0:
            m = train(L[:, :t+1])
            pred = np.array([max(m.predict([L[i, t-K+1:t+1]])[0], 0) for i in range(N_VM)])
            mig += rebalance(a, pred)
        hist.append(a.copy())
    return np.array(hist), mig

def metrics(L, hist, mig):
    hl = np.zeros((T, N_HOST))
    for t in range(T): np.add.at(hl[t], hist[t], L[:, t])
    out = {}
    for name, sl in [("before", slice(0, SHIFT)), ("after", slice(SHIFT, T))]:
        out[name] = dict(balance_std=(hl[sl] / CAP).std(axis=1).mean(),
                         overloads=int((hl[sl] > CAP).sum()))
    out["migrations"] = mig
    return out, hl

if __name__ == "__main__":
    L = gen_trace()
    runs = {"first_fit": run_static(L, first_fit),
            "best_fit": run_static(L, best_fit),
            "predictive": run_predictive(L)}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
    for k, (name, (hist, mig)) in enumerate(runs.items()):
        m, hl = metrics(L, hist, mig); print(name, m)
        ax[k].plot(hl); ax[k].axhline(CAP, c="r", ls="--")
        ax[k].axvline(SHIFT, c="k", ls=":"); ax[k].set_title(name)
    plt.tight_layout(); plt.savefig("results/host_load.png", dpi=150)