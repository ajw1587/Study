# 미래 도시
# 미래 도시에는 1번부터 N번까지의 회사가 있는데 특정 회사끼리는 서로 도로를 통해 연결되어 있단.
# 방문 판매원 A는 현재 1번 회사에 위치 있으며, X번 회사에 방문해 물건을 판매하고자 한다.
# 연결된 2개의 회사는 양방향으로 이동할 수 있다.
# 방문 판매원 A는 K번 회사에 가려고 한다.
# 따라서 A는 1번 회사에서 출발하여 K번 회사를 방문한 뒤에 X번 회사로 가는 것이 목표다.
# 회사 사이를 이동하게 되는 최소 시간을 계산하는 프로그램을 작성하시오.

# 첫째 줄에 전체 회사의 개수 N과 경로의 개수 M이 공백으로 구분되어 라쳬대로 주어진다.
# (1 <= N, M <= 100)
# 둘째 줄부터 M + 1번째 줄에는 연결된 두 회사의 번호가 공백으로 구분되어 주어진다.

INF = int(1e9)

# 노드의 개수 및 간선의 개수를 입력받기
n, m = map(int, input().split())

# 2차원 리스트(그래프 표현)을 만들고, 모든 값을 무한으로 초기화
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에게 가는 비용은 0으로 초기화
for i in range(n + 1):
  for j in range(n + 1):
    if i == j:
      graph[i][j] = 0

# 각 간선에 대한 정보를 입렵 받고, 그 값으로 초기화
for _ in range(m):
  x, y = map(int, input().split())
  graph[x][y] = 1
  graph[y][x] = 1

# 거쳐갈 노드 k와 최종 목적지 노드 x 입력받기
x, k = map(int, input().split())

# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n + 1):
  for a in range(1, n + 1):
    for b in range(1, n + 1):
      graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])
    
# 수행된 결과를 출력
distance = graph[1][k] + graph[k][x]

# 도달할 수 없는 경우, -1을 출력
if distance >= INF:
  print('-1')
else:
  print(distance)