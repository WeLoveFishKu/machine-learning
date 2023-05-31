import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ScrapeDataFunctions:
    def __init__(self):
        pass

    def set_dates(self):
        tgl = datetime.now()
        angkaTgl = tgl.day
        dayName = tgl.strftime("%A")[:3]
        monthName = tgl.strftime("%B")[:3]
        year = tgl.year
        words = f'{dayName} {monthName} {angkaTgl} {year}'
        return words

    def scrape_data(self):
        options = Options()
        options.add_argument("start-maximized")
        webdriver_path = './chromedriver'

        driver = webdriver.Chrome(executable_path=webdriver_path, options=options)
        driver.get("https://panelharga.badanpangan.go.id/harga-eceran")

        # clicks the button to change date ranges
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "control")))
        _ = driver.find_element(By.CLASS_NAME, 'control').click()

        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "datepicker-nav")))
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav').click()
        _ = driver.find_element(By.XPATH, "//div[@data-year='2022']").click()
        _ = driver.find_element(By.XPATH, "//div[@data-date='Sun May 01 2022 00:00:00 GMT+0700 (Western Indonesia Time)']").click()

        words = self.set_dates()

        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "datepicker-nav")))
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav').click()
        _ = driver.find_element(By.XPATH, "//div[@data-year='2023']").click()
        _ = driver.find_element(By.XPATH, f"//div[@data-date='{words} 00:00:00 GMT+0700 (Western Indonesia Time)']").click()

        for i in range(1, 35):
            while True:
                try:
                    _ = driver.find_element(By.XPATH, "//span[@class='select2-selection__rendered']").click()
                    province_html = driver.find_element(By.XPATH, f"//option[@value='{i}']")
                    province = province_html.text
                    _ = province_html.click()
                
                    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table")))
                    table_html = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table"))).get_attribute("outerHTML")
                    df = pd.read_html(table_html)[0]
                    df.to_csv(f'./DataHargaIkan/DataHargaIkan_{province}.csv')
                    break
                except:
                    time.sleep(10)

        driver.quit()

        return