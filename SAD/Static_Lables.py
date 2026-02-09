import numpy as np
from collections import Counter

CACHE_PATH = r"E:\Duc\Safety Driving (2)\Safety Driving\SAD\SAD_cache_hp0.5_lp50.0_fs250_t-1.0_2.0_final_.npz"

data = np.load(CACHE_PATH, allow_pickle=True)

subjects = list(data["subjects"])
subjects_X = list(data["subjects_X"])  # list of (Ni, C, T)
subjects_Y = list(data["subjects_Y"])  # list of (Ni,)

print("========== SAD CACHE SUMMARY ==========")

total_counter = Counter()
total_samples = 0

for s, X, y in zip(subjects, subjects_X, subjects_Y):
    cnt = Counter(y.tolist())
    n = len(y)
    total_samples += n
    total_counter.update(cnt)

    print(
        f"Subject {s:>4s} | "
        f"alert(0)={cnt.get(0,0):5d} | "
        f"drowsy(1)={cnt.get(1,0):5d} | "
        f"total={n:5d} | "
        f"X shape={X.shape}"
    )

print("--------------------------------------")
print(f"TOTAL subjects      : {len(subjects)}")
print(f"TOTAL alert(0)      : {total_counter.get(0,0)}")
print(f"TOTAL drowsy(1)     : {total_counter.get(1,0)}")
print(f"TOTAL samples       : {total_samples}")
print("======================================")
