# n: 일
# v: 일에 따른 판매 금액
# 판 후에 살 수 있다.
# 최대 이익을 남길때의 값은?

n = 10
v = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

answer1 = 0
answer2 = 0
v1 = v[:]
v2 = v[:]
for i in range(len(v)):
    max_value = max(v1)
    min_value = min(v1)
    if v1.index(max_value) < v1.index(min_value):
        answer1 = max_value - min_value
        break
    else:
        v1.pop(v1.index(min_value))

print(answer1)