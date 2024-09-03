from bisect import bisect_left

a = [1, 2, 3, 7, 8]
index = bisect_right(a, 3)
print(index)