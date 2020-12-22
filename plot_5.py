import numpy as np
import matplotlib.pyplot as plt


def sort(arr):
    dis_arr = []
    for t in range(arr.shape[0]):
        t_dis = 0
        for ant in range(arr.shape[1]):
            for ant_2 in range(arr.shape[1]):
                t_dis += dis(arr[t, ant, 0], arr[t, ant, 1], arr[t, ant_2, 0], arr[t, ant_2, 1])
        dis_arr.append(t_dis / arr.shape[1])
    return dis_arr


def dis(x1, y1, x2, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


compare_10 = "imgs/compare_10_locations.npy"
compare_30 = "imgs/compare_30_locations.npy"
compare_50 = "imgs/compare_50_locations.npy"
compare_70 = "imgs/compare_70_locations.npy"


compare_10 = np.load(compare_10)

compare_30 = np.load(compare_30)

compare_50 = np.load(compare_50)

compare_70 = np.load(compare_70)

arr_10 = sort(compare_10)
arr_30 = sort(compare_30)
arr_50 = sort(compare_50)
arr_70 = sort(compare_70)


fig, ax = plt.subplots(1, 1)
ax.plot(arr_10, label="10")
ax.plot(arr_30, label="30")
ax.plot(arr_50, label="50")
ax.plot(arr_70, label="70")
plt.xlabel("Time")
plt.ylabel("Distance coeff")
plt.legend()
plt.savefig("imgs/distance_over_time.png")
# plt.show()

