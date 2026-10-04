import numpy as np
import vm_placement as vp
from shift_types import make_trace

THR = 0.85

def explained_rebalance(a, pred, L, t, log):
    for h in range(vp.N_HOST):
        while pred[a == h].sum() > THR * vp.CAP:
            vms = np.where(a == h)[0]
            if len(vms) <= 1:
                break
            v = vms[np.argmax(pred[vms])]
            tgt = min(range(vp.N_HOST), key=lambda x: pred[a == x].sum())
            if tgt == h or pred[a == tgt].sum() + pred[v] > THR * vp.CAP:
                break

            src_pred = pred[vms].sum()
            tgt_after = pred[a == tgt].sum() + pred[v]

            excess = (src_pred - THR * vp.CAP) / vp.CAP
            conf = float(np.clip(0.5 + 2 * excess, 0.5, 1.0))

            future = L[vms, t + 1: t + vp.H + 1].sum(axis=0)
            correct = bool(future.size > 0 and future.max() > vp.CAP)

            text = (f"t={t}: moved VM {v} from host {h} to host {tgt}. "
                    f"Host {h} predicted load {src_pred:.0f}/{vp.CAP} "
                    f"(limit {THR*vp.CAP:.0f}); host {tgt} would be {tgt_after:.0f} after move.")
            log.append(dict(conf=conf, correct=correct, text=text))
            a[v] = tgt

def run_explained(L):
    a = vp.best_fit(L[:, 0])
    log = []
    for t in range(vp.T):
        if t >= vp.WARM and t % vp.INTERVAL == 0:
            m = vp.train(L[:, :t + 1])
            pred = np.array([max(m.predict([L[i, t - vp.K + 1:t + 1]])[0], 0)
                             for i in range(vp.N_VM)])
            explained_rebalance(a, pred, L, t, log)
    return log

def mean_or_nan(x):
    return float(np.mean(x)) if len(x) else float("nan")

if __name__ == "__main__":
    vp.rng = np.random.default_rng(0)
    log = run_explained(make_trace("ramp"))
    print("Example explanations (seed 0, ramp):")
    for e in log[:5]:
        print(f"  [conf={e['conf']:.2f}, correct={e['correct']}] {e['text']}")
    print()

    print(f"{'shift':<8}{'n':>5}{'accuracy':>10}{'conf|correct':>14}{'conf|wrong':>12}"
          f"{'acc(conf>=.8)':>15}{'acc(conf<.8)':>14}")
    for mode in ["ramp", "jump", "burst"]:
        allv = []
        for s in range(15):
            vp.rng = np.random.default_rng(s)
            allv += run_explained(make_trace(mode))
        conf = np.array([e["conf"] for e in allv])
        ok = np.array([e["correct"] for e in allv])
        hi, lo = conf >= 0.8, conf < 0.8
        print(f"{mode:<8}{len(allv):>5}{ok.mean():>10.2f}"
              f"{mean_or_nan(conf[ok]):>14.2f}{mean_or_nan(conf[~ok]):>12.2f}"
              f"{mean_or_nan(ok[hi]):>15.2f}{mean_or_nan(ok[lo]):>14.2f}")