from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import re
import time
import pandas as pd
from datetime import datetime

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)
df = pd.DataFrame(columns=['Timestamp', 'TS', 'Liq', 'Vol', 'Spread', 'Red', 'Yellow', 'Green', 'NVP', 'COP'])
driver.get('https://www.coingecko.com/en/exchanges/latoken#trust_score')
while True:
    time.sleep(3)

    t = datetime.now()

    ts = driver.find_element(
        by=By.XPATH,
        value='//*[@id="trust_score"]/div[1]/div/div[1]/div[2]'
    ).get_attribute("innerText")

    liq = driver.find_element(
        by=By.XPATH,
        value='//*[@id="trust_score"]/div[1]/div/table[1]/tbody/tr/td[1]'
    ).get_attribute("innerText")

    vol = re.sub(
        "[^0-9]", "", driver.find_element(
            by=By.XPATH,
            value='//*[@id="trust_score"]/div[2]/div/div[1]/div/table/tbody/tr[1]/td/div'
        ).get_attribute("innerText")
    )

    spread = (lambda x: float(x.get_attribute("innerText")[:-1]) / 100)(
        driver.find_element(
            by=By.XPATH,
            value='//*[@id="trust_score"]/div[2]/div/div[1]/div/table/tbody/tr[4]/td'
        )
    )

    bar = driver.find_element(
        by=By.XPATH,
        value='//*[@id="trust_score"]/div[2]/div/div[1]/div/table/tbody/tr[5]/td/div'
    )

    green, yellow, red = [
        re.findall('\d+\%', x.get_attribute("style"))[0] for x in bar.find_elements(By.XPATH, ".//*")[:-1]
    ]

    nvp = driver.find_element(
        by=By.XPATH,
        value='//*[@id="trust_score"]/div[2]/div/div[2]/div/table/tbody/tr[1]/td'
    ).get_attribute("innerText")[:-2]

    cop = driver.find_element(
        by=By.XPATH,
        value='//*[@id="trust_score"]/div[2]/div/div[2]/div/table/tbody/tr[2]/td'
    ).get_attribute("innerText")[:-2]

    df.loc[len(df.index)] = [t, ts, liq, vol, spread, red, yellow, green, nvp, cop]
    df.set_index('Timestamp').to_excel('Rankings.xlsx')
    print(df)
    time.sleep(257)
    driver.refresh()