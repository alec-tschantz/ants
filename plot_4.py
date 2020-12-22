import numpy as np 
import matplotlib.pyplot as plt 

compare_10 = "imgs/compare_10_round_trips.npy"
compare_30 = "imgs/compare_30_round_trips.npy"
compare_50 = "imgs/compare_50_round_trips.npy"
compare_70 = "imgs/compare_70_round_trips.npy"


compare_10 = np.load(compare_10)
compare_10 = compare_10 * 10

compare_30 = np.load(compare_30)
compare_30 = compare_30 * 30

compare_50 = np.load(compare_50)
compare_50 = compare_50 * 50

compare_70 = np.load(compare_70)
compare_70 = compare_70 * 70

fig, ax = plt.subplots(1, 1)
ax.plot(compare_10, label="10")
ax.plot(compare_30, label="30")
ax.plot(compare_50, label="50")
ax.plot(compare_70, label="70")
plt.xlabel("Time")
plt.ylabel("Number of round trips")
plt.legend()
plt.savefig("imgs/round_trips_not_norm.png")