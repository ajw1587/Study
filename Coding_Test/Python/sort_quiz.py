# N개의 원소로 구성되어있는 A, B 두개의 배열이 있다.
# K번 바꿔치기 연산을 이용해서 배열 A 원소들의 합이 최대가 되도록 해라.

n, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a = a.sort()
b = b.sort(reverse = True)

# 첫번째 인덱스부터 확인
for i in range(k):
    if a[i] < b[i]:
        a[i], b[i] = b[i], a[i]
    else:
        break

print(sum(a))
