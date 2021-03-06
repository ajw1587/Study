# 효율적인 화폐 구성
# N가지 종류의 화폐가 있다.
# 이 화폐들의 개수를 최소한으로 이용해서 그 가치의 합이 M원이 되도록 하려 한다.
# 이때 각 종류의 화폐는 몇 개라도 사용할 수 있다.
# M원을 만들기 위한 최소한의 화폐 개수를 출력하는 프로그램을 작성하라.
# 첫째 줄에 N, M이 주어진다. 단, 불가능할때는 -1 출력
# (1 <= N <= 100, 1 <= M <= 10000)

n, m = map(int, input().split())
n_list = []
for i in range(n):
    x = int(input())
    n_list.append(x)

m_list = [10001] * (m + 1)
m_list[0] = 0
for i in range(n):
    for j in range(n_list[i], m + 1):
        if m_list[j - n_list[i]] != 10001:
            m_list[j] = min(m_list[j - n_list[i]] + 1, m_list[j])

if m_list[m] == 10001:
    print(-1)
else:
    print(m_list[m])