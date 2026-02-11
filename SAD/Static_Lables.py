import numpy as np
from collections import Counter

CACHE_PATH = r"C:\Users\User\Desktop\HPFML-IC-2--ASTER\SAD\SAD_cache_paper_hp1.0_lp50.0_fs128_t-3.0_0.0_aug1x1_minAfterAug0.npz"

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
