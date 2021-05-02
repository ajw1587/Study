# https://www.youtube.com/watch?v=jjOmN2_lmdk&list=PLVsNizTWUw7H9_of5YCB0FmsSc-K44y81&index=27
# 절단기에 높이(H)를 지정하면 줄지어진 떡을 한 번에 절단 합니다.
# 높이가 H보다 긴 떡은 H 위의 부분이 잘리고, 낮은 떡은 잘리지 않습니다.
# 예를 들어 19, 14, 10, 17 cm 인 떡이 나란히 있고 절단기 높이를 15cm로 지정하면
# 자른 뒤 떡의 높이는 15, 14, 10, 15cm가 될 것입니다.
# 잘린 떡의 길이는 4, 0, 0, 2cm 입니다.
# 손님은 6cm 만큼의 길이를 가져갑니다.
# 손님이 왔을 때 요청한 총 길이가 M일 때 적어도 M만큼의 떡을 얻기 위해
# 절단기에 설정할 수 있는 높이의 최댓값을 구하는 프로그램을 작성하세요.

# 떡의 개수 N, 요청한 떡의 길이 M
n, m = map(int, input().split())
# 떡들의 길이
h = list(map(int, input().split()))

start = 0
end = max(h)

result = 0
while(start <= end):
    total = 0
    mid = (start + end) // 2
    for x in h:
        if x > mid:
            total += x - mid
    if total < m:
        end = mid - 1
    else:
        result = total
        start = mid + 1
print(result)