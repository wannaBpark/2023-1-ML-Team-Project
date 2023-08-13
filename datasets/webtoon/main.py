import os
import requests
from bs4 import BeautifulSoup

# 웹툰 URL의 기본 부분
base_url = "https://comic.naver.com/webtoon/detail?titleId=676695&no="

# 이미지 다운로드 폴더 생성
if not os.path.exists('webtoon_images'):
    os.makedirs('webtoon_images')

# 각 에피소드에 대해
for episode in range(1, 345):
    # 웹툰 URL
    url = base_url + str(episode)

    # 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 이미지 태그 가져오기
    img_tags = soup.select('div.wt_viewer img')

    # 이미지 다운로드
    for i, img_tag in enumerate(img_tags):
        img_url = img_tag['src']
        headers = {'Referer': url}
        response = requests.get(img_url, headers=headers)
        with open('webtoon_images/episode_{}_img_{}.jpg'.format(episode, i), 'wb') as f:
            f.write(response.content)
