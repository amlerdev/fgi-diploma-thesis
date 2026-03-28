from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

print("Načítám stránku...")
driver.get("https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/")
time.sleep(10)

data = driver.execute_script("""
    return new Promise((resolve) => {
        window.finhackerDataPromise.then(data => resolve(data));
    });
""")

driver.quit()

# Použít 'daily' pro denní data
df_daily = pd.DataFrame(data['daily'])
print(f"\nDaily data: {len(df_daily)} záznamů")
print(df_daily.head())

# Filtrovat od 2011-01-01
df_daily['date'] = pd.to_datetime(df_daily.iloc[:, 0])
df_daily = df_daily[df_daily['date'] >= '2011-01-01']

df_daily.to_csv('fear_greed_historical.csv', index=False)
print(f"\nUloženo {len(df_daily)} záznamů od 2011-01-01")
print(df_daily.tail())