# N x M 크기의 직사각형 형태의 미로에 갇혔다.
# 미로에는 여러마리의 괴물이 있다.
# 사용자의 위치는 (1, 1)이며 미로의 출구는 (N, M)의 위치에 존재하며 한번에 한칸씩 이동할 수 있다.
# 괴물이 있는 부분은 0, 괴물이 없는 부분은 1로 표시되어 있습니다.
# 미로는 반드시 탈출할 수 있는 형태로 제시
# 탈출하기 위해 움직여야 하는 최소 칸의 개수는?
from collections import deque

def bfs(x, y):
    queue = deque()
    queue.append((x, y))

    # 이동할 네 가지 방향 정의 (상, 하, 좌, 우)
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    # 큐가 빌때까지 반복
    while queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            # 해당 노드를 처음 방문하는 경우에만 최단거리 기록
            if graph[nx][ny] == 1:
                graph[nx][ny] = graph[x][y] + 1
                queue.append((nx, ny))
    # 가장 오른쪽 아래까지의 최단 거리 반환
    return graph[n-1][m-1]

n, m = map(int, input().split())
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# BFS를 수행한 결과 출력
print(bfs(0, 0))