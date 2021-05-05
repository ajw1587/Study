from PIL import Image, ImageOps, ImageFilter

img = Image.open('F:/paper_texture.png')
deco = Image.open('F:/example_noise2.png')

deco = deco.convert("RGBA")
datas = deco.getdata()

newData = []
cutOff = 200
for item in datas:
    if item[0] >= cutOff and item[1] >= cutOff and item[2] >= cutOff:
        newData.append((255, 255, 255, 0))
        # RGB의 각 요소가 모두 cutOff 이상이면 transparent하게 바꿔줍니다.
    else:
        newData.append(item)
        # 나머지 요소는 변경하지 않습니다.
 
deco.putdata(newData)

img.paste(deco)
img.save('F:/AAAAA.png')