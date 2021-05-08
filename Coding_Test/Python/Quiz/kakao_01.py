
def solution(s):
    dic = {'zero':'0', 'one':'1', 'two':'2', 'three':'3', 'four':'4', 'five':'5',
           'six':'6', 'seven':'7', 'eight':'8', 'nine':'9'}
    
    num_array = ''
    str_array = ''
    for i in s:
        if ord(i) >= 48 and ord(i) <= 57:
            num_array = num_array + i
            continue
        else:
            str_array = str_array + i
            for j, k in dic.items():
                if str_array == j:
                    num_array = num_array + k
                    str_array = ''
            continue
    answer = int(num_array)
    return answer