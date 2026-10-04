# Predictive VM Placement

Compares static VM placement (first-fit, best-fit, best-fit with 80% limit)
against predictive placement (LinearRegression forecast + migration) under
three mid-trace load shifts: ramp, jump, burst.

## Setup
pip install numpy scikit-learn matplotlib
mkdir results

## How to run
python vm_placement.py      # single run + results/host_load.png
python multi_seed.py        # 15-seed comparison
python sensitivity.py       # K, H sensitivity
python shift_types.py       # ramp / jump / burst, 3 policies
python strong_baseline.py   # adds best-fit with 80% limit
python sliding_window.py    # sliding-window retraining
python explain.py           # explanation confidence

## Results summary
After the shift, predictive placement has far fewer overloads than static
policies (ramp: 0.3 vs 192.3; jump: 5.2 vs 239.9; burst: 1.9 vs 102.1) at a
cost of about 6-7 migrations. An 80% fill limit helps static placement
somewhat but not under a permanent jump. Sliding-window retraining did not
improve results. In the bonus, higher-confidence migrations were more often
correct (accuracy 0.82-0.91 for conf >= 0.8 vs 0.43-0.48 below).
First-fit and best-fit gave identical results in all seeds.

## AI assistance disclosure
I used Claude (Anthropic) for implementation help, for discussing experiment
ideas and for help drafting the report and README text. I ran all
experiments myself and reviewed the results.
