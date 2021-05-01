# N x M 의 얼음틀이 있다.
# 뚫린 부분은 0, 벽이 있는 부분은 1
# 여기서 아이스크림이 생성되는 수는???
# ex)
# 0 0 1 1 0 0
# 0 0 1 1 1 0
# 1 1 1 1 0 0
# 0 0 1 0 0 0
# 0 0 1 0 0 0

def dfs(x, y):
    # 범위를 벗어나는 경우
    if x < -1 or y < -1 or x >= n or y >= m:
        return False
    # 현재 노드를 아직 방문하지 않았다면
    if graph[x][y] == 0:
        graph[x][y] = 1

        dfs(x-1, y)
        dfs(x+1, y)
        dfs(x, y-1)
        dfs(x, y+1)
        return True
    return False

n, m = map(int, input().split())

graph = []
for i in range(n):
    graph.append(list(map(int, input())))

result = 0
for i in range(n):
    for j in range(m):
        if dfs(i, j) == True:
            result += 1
print(result)