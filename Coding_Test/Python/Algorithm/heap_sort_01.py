# 최소힙
import heapq

# 오름차순 힙 정렬(Heap Sort)
def heap_sort(iterable):
  h = []
  result = []
  # 모든 원소를 차레대로 힙에 삽입
  for value in iterable:
    heapq.heappush(h, value)
  # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
  for i in range(len(h)):
    result.append(heapq.heappop(h))
  return result

result = heap_sort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
print(result)


# 최대힙: 파이썬은 최소힙은 제공하지만 최대힙은 제공하지 않기 때문에 -를 붙여준다.
# 내림차순 힙 정렬(Heap Sort)
def heap_sort(iterable):
  h = []
  result = []
  # 모든 원소를 차레대로 힙에 삽입
  for value in iterable:
    heapq.heappush(h, -value)
  # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
  for i in range(len(h)):
    result.append(-heapq.heappop(h))
  return result

result = heap_sort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
print(result)