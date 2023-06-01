from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import requests

import time

def scrape_data():
    options = Options()
    options.add_argument("start-maximized")
    webdriver_path = './chromedriver'
    driver = webdriver.Chrome(executable_path=webdriver_path, options=options)
    # url_pasarikan = "https://www.google.com/search?q=ikan+dijual+di+pasar&tbm=isch&ved=2ahUKEwjdquGNnYX_AhX-itgFHUpSAuYQ2-cCegQIABAA&oq=ikan+dijual+di+pasar&gs_lcp=CgNpbWcQAzIHCAAQExCABDIICAAQBRAeEBMyCAgAEAgQHhATMggIABAIEB4QEzIICAAQCBAeEBMyCAgAEAgQHhATMggIABAIEB4QEzIICAAQCBAeEBMyCAgAEAgQHhATMggIABAIEB4QEzoECCMQJzoFCAAQgAQ6BggAEAcQHjoICAAQBRAHEB46CAgAEAgQBxAeOgYIABAIEB5QtQVYqQ1gtQ9oAHAAeACAAXaIAcAGkgEDMy41mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=gW9pZJ3mOP6V4t4PyqSJsA4&bih=970&biw=1680"
    url_pasar = "https://www.google.com/search?q=jualan+sayur+dan+buah+di+pasar&tbm=isch&ved=2ahUKEwiw6K_MnoX_AhWei9gFHaRMAdEQ2-cCegQIABAA&oq=jualan+sayur+dan+buah+di+pasar&gs_lcp=CgNpbWcQAzoECCMQJzoHCCMQ6gIQJzoICAAQgAQQsQM6BQgAEIAEOgUIABCxAzoECAAQHjoGCAAQCBAeOggIABCxAxCDAToECAAQAzoHCAAQigUQQzoGCAAQBRAeUJYGWIepK2CWqitoDHAAeACAAbEBiAGHIZIBBTI3LjE2mAEAoAEBqgELZ3dzLXdpei1pbWewAQrAAQE&sclient=img&ei=EXFpZPCbKZ6X4t4PpJmFiA0&bih=970&biw=1680"
    # driver.get(url_pasarikan)
    driver.get(url_pasar)
    store_urls = []

    for i in range(300):
        while True:
            try:
                if driver.find_element(By.XPATH, f"//div[@data-ri='{i}']").get_attribute('class') == 'isv-r PNCib iPukc J3Tg1d':
                    break
                else:
                    _ = driver.find_element(By.XPATH, f"//div[@data-ri='{i}']").click()
                    url = driver.find_element(By.XPATH, "//img[@jsaction='VQAsE']").get_attribute('src')
                    store_urls.append(url)
                    break
            except:
                time.sleep(10)
    driver.quit()

    return store_urls

def download_base64_image(url, output_file):
    encoded_data = url.split(',')[1]
    image_data = base64.b64decode(encoded_data)
    with open(output_file, 'wb') as file:
        file.write(image_data)

    return

def download_image(url, output_file):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except:
        pass

    return

store_urls = scrape_data()

for i, url in enumerate(store_urls):
    # output_file = f"fish_image{i}.jpg"
    output_file = f'not_fish_image{i}.jpg'
    try:
        download_base64_image(url, output_file)
    except:
        download_image(url, output_file)
