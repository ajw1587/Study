from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

# Chrome Driver Open
driver = webdriver.Chrome(executable_path = 'c:\chromedriver')
# Open "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl")
# Select Search Box in "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"
elem = driver.find_element_by_name("q")
# Put Text in elem(Search Box)
elem.send_keys("sign language one")
# Click Enter(RETURN)
elem.send_keys(Keys.RETURN)
'''
# 스크롤 내리기
SCROLL_PAUSE_TIME = 1
# java script 언어로 현재 스크롤 높이 구함
last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    # Scroll down to bottom
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script('return document.body.scrollHeight')
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
        except:
            break
    last_height = new_height
'''
# 검색창 작은 사진 목록 class 받아오기
images = driver.find_elements_by_css_selector('.rg_i.Q4LuWd')
count = 1
for image in images:
    # Click
    image.click()
    # Loading time
    time.sleep(2)
    # Large Image Get src -> Return: image url
    imgUrl = driver.find_element_by_css_selector('.n3VNCb').get_attribute('src')
    # Save Image
    urllib.request.urlretrieve(imgUrl, 'c:\Study\crawling/inwooinwoo_' + str(count) + '.jpg')
    count += 1

# Close Chrome
# driver.close()
