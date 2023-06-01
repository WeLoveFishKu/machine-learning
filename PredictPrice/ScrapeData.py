import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ScrapeDataFunctions:
    def __init__(self, start_day, start_month, start_year):
        dates = [datetime.strptime(f'{start_day}/{start_month}/{start_year}', '%d/%m/%Y'), 
                 datetime.now()]
        final_dates = []
        for date in dates:
            angkaTgl = date.day
            if len(str(angkaTgl)) == 1:
                angkaTgl = f'0{angkaTgl}'
            dayName = date.strftime("%A")[:3]
            monthName = date.strftime("%B")[:3]
            year = date.year
            final_dates.append(f'{dayName} {monthName} {angkaTgl} {year}')

        self.start_date_year = dates[0].year
        self.today_date_year = dates[1].year
        self.start_date_month = f'0{dates[0].month}' if len(str(dates[0].month)) == 1 else dates[0].month
        self.today_date_month = f'0{dates[1].month}' if len(str(dates[1].month)) == 1 else dates[1].month
        self.start_date_word = final_dates[0]
        self.today_date_word = final_dates[1]

    def scrape_data(self):
        options = Options()
        options.add_argument("start-maximized")
        webdriver_path = './chromedriver'
        driver = webdriver.Chrome(executable_path=webdriver_path, options=options)
        driver.get("https://panelharga.badanpangan.go.id/harga-eceran")
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "control")))
        _ = driver.find_element(By.CLASS_NAME, 'control').click()
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "datepicker-nav")))
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav-year').click()
        _ = driver.find_element(By.XPATH, f"//div[@data-year='{self.start_date_year}']").click()
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav-month').click()
        _ = driver.find_element(By.XPATH, f"//div[@data-month='{self.start_date_month}']").click()
        _ = driver.find_element(By.XPATH, f"//div[@data-date='{self.start_date_word} 00:00:00 GMT+0700 (Western Indonesia Time)']").click()
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "datepicker-nav")))
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav-year').click()
        _ = driver.find_element(By.XPATH, f"//div[@data-year='{self.today_date_year}']").click()
        _ = driver.find_element(By.CLASS_NAME, 'datepicker-nav-month').click()
        _ = driver.find_element(By.XPATH, f"//div[@data-month='{self.today_date_month}']").click()
        _ = driver.find_element(By.XPATH, f"//div[@data-date='{self.today_date_word} 00:00:00 GMT+0700 (Western Indonesia Time)']").click()
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
                    df.to_csv(f'./DataHargaIkan_{province}.csv')
                    break
                except:
                    time.sleep(10)
        driver.quit()

        return