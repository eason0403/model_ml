import requests
import json
import pandas as pd
import os
from bs4 import BeautifulSoup

# 爬取主客隊名稱和比賽時間
def get_game_info(game_id):
    url = f'https://pleagueofficial.com/game/{game_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # 獲取主客隊名稱
    away_raw = soup.find('div', class_='col-lg-7 col-12 text-right align-self-center').text.strip()
    home_raw = soup.find('div', class_='col-lg-7 col-12 text-left align-self-center').text.strip()

    away = away_raw.split('\n')[4].strip()
    home = home_raw.split('\n')[4].strip()

    # 獲取比賽時間
    game_time_raw = soup.find("span", class_="fs14 text-white").text.strip()
    game_time = game_time_raw.split('\n')[1].strip()
    
    return home, away, game_time

# 獲取比賽數據的函數
def single_game_data(game_id):
    parameters = {
        "id": game_id,
        'away_tab': 'total',
        'home_tab': 'total'
    }
    response = requests.get("https://pleagueofficial.com/api/boxscore.php?", params=parameters)
    return response.text

# 解析 JSON 並保存資料
def toJson(game_id, output_file='all_games_data.csv'):
    home_team, away_team, game_time = get_game_info(game_id)  # 獲取主客隊名稱和比賽時間
    data = single_game_data(game_id)

    # 解析 JSON 字串
    parsed_data = json.loads(data)

    # 檢查是否有錯誤
    if parsed_data['error'] == "":
        game_data = parsed_data['data']
        
        # 提取主隊和客隊數據
        home_players = game_data['home']
        away_players = game_data['away']
        
        # 主隊和客隊總數據
        home_total = game_data['home_total']
        away_total = game_data['away_total']

        # 將主隊數據轉換為 DataFrame
        home_df = pd.DataFrame(home_players)
        home_df['team'] = 'Home'  # 添加一列來標記球隊
        home_df['team_name'] = home_team  # 添加主隊名稱
        home_df['game_time'] = game_time  # 添加比賽時間

        # 將客隊數據轉換為 DataFrame
        away_df = pd.DataFrame(away_players)
        away_df['team'] = 'Away'  # 添加一列來標記球隊
        away_df['team_name'] = away_team  # 添加客隊名稱
        away_df['game_time'] = game_time  # 添加比賽時間

        # 合併主隊和客隊數據
        combined_df = pd.concat([home_df, away_df], ignore_index=True)

        # 建立全隊總數據的 DataFrame
        team_totals = pd.DataFrame([{
            'game_id':game_id,
            'name': home_team,
            'is_home': 1,  # 主場設為 1
            'two_m': home_total['two_m'],
            'two': home_total['two'],
            'twop': home_total['twop'],
            'trey_m': home_total['trey_m'],
            'trey': home_total['trey'],
            'treyp': home_total['treyp'],
            'ft_m': home_total['ft_m'],
            'ft': home_total['ft'],
            'ftp': home_total['ftp'],
            'points': home_total['points'],
            'reb': home_total['reb'],
            'reb_o': home_total['reb_o'],
            'reb_d': home_total['reb_d'],
            'ast': home_total['ast'],
            'stl': home_total['stl'],
            'blk': home_total['blk'],
            'turnover': home_total['turnover'],
            'pfoul': home_total['pfoul'],
        }, {
            'game_id':game_id,
            'name': away_team,
            'is_home': 0,
            'two_m': away_total['two_m'],
            'two': away_total['two'],
            'twop': away_total['twop'],
            'trey_m': away_total['trey_m'],
            'trey': away_total['trey'],
            'treyp': away_total['treyp'],
            'ft_m': away_total['ft_m'],
            'ft': away_total['ft'],
            'ftp': away_total['ftp'],
            'points': away_total['points'],
            'reb': away_total['reb'],
            'reb_o': away_total['reb_o'],
            'reb_d': away_total['reb_d'],
            'ast': away_total['ast'],
            'stl': away_total['stl'],
            'blk': away_total['blk'],
            'turnover': away_total['turnover'],
            'pfoul': away_total['pfoul'],
        }])

        # 合併球員數據和全隊總數據
        # combined_df = pd.concat([combined_df, team_totals], ignore_index=True)
        # 全隊數據
        combined_df = pd.DataFrame(team_totals)
        if not os.path.isfile(output_file):
            # 如果檔案不存在，寫入資料並包含標題
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"成功創建並保存數據到 {output_file}")
        else:
            # 如果檔案已存在，附加數據，且不寫入標題
            combined_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"成功附加數據到 {output_file}")

    else:
        print(f"發生錯誤: {parsed_data['error']}")

# 獲取並保存多場比賽的數據
for i in range(598, 605):
    toJson(i)