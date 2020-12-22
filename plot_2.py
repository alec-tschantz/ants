import numpy as np

import matplotlib.pyplot as plt


round_trips_10 = [1, 3, 0, 3, 3, 4, 2]
round_trips_30 = [13, 12, 9, 6, 9]
round_trips_50 = [17, 15, 22, 14, 12, 14]
round_trips_70 = [21, 36, 31, 35]


norm_round_trips_10 = [el / 10 for el in round_trips_10]
norm_round_trips_30 = [el / 30 for el in round_trips_30]
norm_round_trips_50 = [el / 50 for el in round_trips_50]
norm_round_trips_70 = [el / 70 for el in round_trips_70]

mean_10 = np.mean(np.array(norm_round_trips_10))
mean_30 = np.mean(np.array(norm_round_trips_30))
mean_50 = np.mean(np.array(norm_round_trips_50))
mean_70 = np.mean(np.array(norm_round_trips_70))

coeffs_10 = [277143, 215814, 286388, 249350, 196007, 232926, 230406]
coeffs_30 = [708663, 704696, 671895, 708203, 715319]
coeffs_50 = [1211381, 1303375, 1271124, 1258744, 1239051, 1358398]
coeffs_70 = [1745315, 1706919, 1741642, 1842879]

coeff_10 = np.mean(np.array(coeffs_10))
coeffs_30 = np.mean(np.array(coeffs_30))
coeffs_50 = np.mean(np.array(coeffs_50))
coeffs_70 = np.mean(np.array(coeffs_70))

plt.bar(
    range(4),
    [mean_10, mean_30, mean_50, mean_70],
    color=["r", "g", "b", "y"],
    alpha=0.7,
    tick_label=["10", "30, ", "50", "70"],
)
plt.ylabel("Number of round trips")
plt.xlabel("Number of ants")
plt.savefig("imgs/round_trips.png")
plt.close("all")

plt.plot(range(4), [mean_10, mean_30, mean_50, mean_70], "ro-")
plt.ylabel("Number of round trips")
plt.xlabel("Number of ants")
plt.xticks(range(4), ("10", "30", "50", "70"))
plt.savefig("imgs/round_trips_line.png")
plt.close("all")

plt.plot(range(4), [coeff_10, coeffs_30, coeffs_50, coeffs_70], "ro-")
plt.ylabel("Distance coefficient")
plt.xlabel("Number of ants")
plt.xticks(range(4), ("10", "30", "50", "70"))
plt.savefig("imgs/coeff_line.png")

