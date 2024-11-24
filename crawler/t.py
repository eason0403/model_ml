from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# CSV 檔案名稱
csv_filename = "box_score.csv"

# 設定 ChromeOptions 為 headless 模式
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")
driver_path = r'E:\程式教學\123\chromedriver.exe'

def a(game_id):
    # 為每個執行緒建立獨立的 WebDriver
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f'https://tpbl.basketball/schedule/{game_id}/box-score'
    driver.get(url)

    try:
        # 使用 WebDriverWait 等待表格元素可見
        wait = WebDriverWait(driver, 20)
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.vgt-table.bordered')))

        # 抓取所有表格（主場與客場）
        tables = driver.find_elements(By.CSS_SELECTOR, '.vgt-table.bordered')

        if len(tables) == 2:
            print(f"Game ID {game_id}: 檢測到兩個表格，分別處理客場與主場表格。")

            home_table_html = tables[0].get_attribute('outerHTML')
            away_table_html = tables[1].get_attribute('outerHTML')
        else:
            print(f"Game ID {game_id}: 未檢測到兩個表格，跳過此場比賽。")
            driver.quit()
            return

    except Exception as e:
        print(f"Game ID {game_id} 出現錯誤：{e}")
        driver.quit()
        return

    # 使用 BeautifulSoup 解析 HTML
    away_soup = BeautifulSoup(away_table_html, 'html.parser')
    home_soup = BeautifulSoup(home_table_html, 'html.parser')

    def parse_table(soup, game_id, is_home):
        rows = soup.find_all('tr')
        data = []

        for row in rows:
            cols = row.find_all(['th', 'td'])
            cols = [col.get_text(strip=True) for col in cols]
            data.append(cols)

        df = pd.DataFrame(data)

        def remove_percentage(value):
            if isinstance(value, str) and '%' in value:
                return float(value.replace('%', ''))
            return value

        formatted_data = pd.DataFrame({
            'game_id': [game_id],
            'name': [df.iloc[-2, 0]],
            'is_home': is_home,
            'two_m': [df.iloc[-1, 3]],
            'two': [df.iloc[-1, 4]],
            'twop': [remove_percentage(df.iloc[-1, 5])],
            'trey_m': [df.iloc[-1, 6]],
            'trey': [df.iloc[-1, 7]],
            'treyp': [remove_percentage(df.iloc[-1, 8])],
            'ft_m': [df.iloc[-1, 9]],
            'ft': [df.iloc[-1, 10]],
            'ftp': [remove_percentage(df.iloc[-1, 11])],
            'points': [df.iloc[-1, 2]],
            'reb': [df.iloc[-1, 14]],
            'reb_o': [df.iloc[-1, 12]],
            'reb_d': [df.iloc[-1, 13]],
            'ast': [df.iloc[-1, 15]],
            'stl': [df.iloc[-1, 16]],
            'blk': [df.iloc[-1, 17]],
            'turnover': [df.iloc[-1, 18]],
            'pfoul': [df.iloc[-1, 19]],
        })

        return formatted_data

    away_data = parse_table(away_soup, game_id, is_home=0)
    home_data = parse_table(home_soup, game_id, is_home=1)

    final_data = pd.concat([home_data, away_data], ignore_index=True)

    file_exists = os.path.isfile(csv_filename)
    final_data.to_csv(csv_filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    print(f"成功將 Game ID {game_id} 的資料附加至 CSV 檔案：{csv_filename}")

    # 關閉 WebDriver
    driver.quit()

# 使用 ThreadPoolExecutor 並行處理，設定 max_workers 為 2
game_ids = list(range(9, 26))
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(a, game_ids)
