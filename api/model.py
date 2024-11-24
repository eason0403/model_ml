import json
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from difflib import get_close_matches
import pandas as pd
import joblib

# =====================
# 1. 球隊名稱映射系統
# =====================
team_name_mapping = {
    # P聯盟球隊 - 全名對照
    "桃園璞園領航猿": "桃園璞園領航猿",
    "臺北富邦勇士": "臺北富邦勇士",
    "高雄17直播鋼鐵人": "高雄鋼鐵人",  # 修改這裡
    "臺南台鋼獵鷹": "台鋼獵鷹",  # 修改這裡
    
    # P聯盟球隊 - 簡稱對照
    "領航猿": "桃園璞園領航猿",
    "勇士": "臺北富邦勇士",
    "鋼鐵人": "高雄鋼鐵人",
    "獵鷹": "臺南台鋼獵鷹", 
    
    # T1聯盟球隊 - 全名對照
    "福爾摩沙夢想家": "福爾摩沙夢想家",
    "新北國王": "新北國王",
    "新竹御嵿攻城獅": "新竹御嵿攻城獅",
    "高雄全家海神": "高雄全家海神",
    "新北中信特攻": "新北中信特攻",
    "桃園台啤永豐雲豹": "桃園台啤永豐雲豹",
    "臺北台新戰神": "臺北台新戰神",
    
    # T1聯盟球隊 - 簡稱對照
    "夢想家": "福爾摩沙夢想家",
    "國王": "新北國王",
    "攻城獅": "新竹御嵿攻城獅",
    "海神": "高雄全家海神",
    "戰神": "臺北台新戰神",
    "特攻": "新北中信特攻",
    "雲豹": "桃園台啤永豐雲豹",
    
    # T1歷史名稱對照
    "夢想家TEAM": "福爾摩沙夢想家",
    "特攻TEAM": "新北中信特攻", 
    "海神TEAM": "高雄全家海神",
    "國王TEAM": "新北國王",
    "戰神TEAM": "臺北台新戰神",
    "攻城獅TEAM": "新竹御嵿攻城獅",
    "雲豹TEAM": "桃園台啤永豐雲豹",
    
    # 處理名稱變更歷史
    "臺北戰神": "臺北台新戰神",
    "台啤永豐雲豹": "桃園台啤永豐雲豹",
    "新竹御頂攻城獅": "新竹御嵿攻城獅"
}

# =====================
# 2. 名稱標準化函數
# =====================
def get_standardized_team_name(team_name, season=None):
    """改進的球隊名稱標準化函數"""
    # 首先檢查是否需要特殊處理
    if season and season >= "24-25":
        special_cases = {
            "臺北戰神": "臺北台新戰神",
            "台啤永豐雲豹": "桃園台啤永豐雲豹",
            "新竹御頂攻城獅": "新竹御嵿攻城獅"
        }
        if team_name in special_cases:
            return special_cases[team_name]
    
    # 使用映射表進行轉換
    standardized_name = team_name_mapping.get(team_name, team_name)
    
    # 添加除錯信息
    # print(f"原始球隊名稱: {team_name} -> 標準化後: {standardized_name}")
    
    return standardized_name

def get_team_shortname(team_name):
    """根據球隊全名獲取簡稱"""
    reverse_mapping = {
        "桃園璞園領航猿": "領航猿",
        "臺北富邦勇士": "勇士",
        "高雄17直播鋼鐵人": "鋼鐵人",
        "臺南台鋼獵鷹": "獵鷹",
        "福爾摩沙夢想家": "夢想家",
        "新北國王": "國王",
        "新竹御嵿攻城獅": "攻城獅",
        "高雄全家海神": "海神",
        "臺北台新戰神": "戰神",
        "新北中信特攻": "特攻",
        "桃園台啤永豐雲豹": "雲豹",
        "臺北台新戰神": "戰神"
    }
    std_name = get_standardized_team_name(team_name)
    return reverse_mapping.get(std_name, std_name)

def is_same_team(team_name1, team_name2, season=None):
    """判斷兩個球隊名稱是否為同一支球隊"""
    return get_standardized_team_name(team_name1, season) == get_standardized_team_name(team_name2, season)

# =====================
# 3. 資料讀取和處理
# =====================
def load_json(filename):
    """讀取JSON數據函數"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON in {filename}. Skipping.")
        return []

def load_game_history():
    """讀取比賽歷史數據"""
    try:
        # 讀取P聯盟數據，明確指定列名
        games_df_p = pd.read_csv('../dataset/p_currentgame.csv', names=[
            'game_id', 'name', 'is_home', 'two_m', 'two', 'twop', 
            'trey_m', 'trey', 'treyp', 'ft_m', 'ft', 'ftp', 
            'points', 'reb', 'reb_o', 'reb_d', 'ast', 'stl', 
            'blk', 'turnover', 'pfoul'
        ])
        #print("P聯盟數據讀取完成，形狀:", games_df_p.shape)
        
        # 讀取T1數據，使用相同的列名
        t1_data = pd.read_csv('../dataset/t_currentgame.csv', names=[
            'game_id', 'name', 'is_home', 'two_m', 'two', 'twop', 
            'trey_m', 'trey', 'treyp', 'ft_m', 'ft', 'ftp', 
            'points', 'reb', 'reb_o', 'reb_d', 'ast', 'stl', 
            'blk', 'turnover', 'pfoul'
        ])
        #print("T1數據讀取完成，形狀:", t1_data.shape)
        
        # 檢查列名是否正確設置
        #print("P聯盟數據列名:", games_df_p.columns.tolist())
        #print("T1數據列名:", t1_data.columns.tolist())
        
        # 標準化T1的隊伍名稱
        t1_name_mapping = {
            "夢想家TEAM": "福爾摩沙夢想家",
            "特攻TEAM": "新北中信特攻",
            "海神TEAM": "高雄全家海神",
            "國王TEAM": "新北國王",
            "戰神TEAM": "臺北台新戰神",
            "攻城獅TEAM": "新竹御嵿攻城獅",
            "雲豹TEAM": "桃園台啤永豐雲豹"
        }
        t1_data['name'] = t1_data['name'].map(lambda x: t1_name_mapping.get(x, x))
        
        # 合併數據
        games_df = pd.concat([games_df_p, t1_data], ignore_index=True)
        #print("合併後的數據形狀:", games_df.shape)
        
        # 確認合併後的列名
        #print("合併後的列名:", games_df.columns.tolist())
        
        # 為每場比賽添加對手得分
        games_df['opponent_points'] = games_df.groupby('game_id')['points'].transform(lambda x: x.iloc[::-1].values)
        
        # 打印所有出現的球隊名稱，用於除錯
        #print("數據中出現的所有球隊名稱:", sorted(games_df['name'].unique()))
        
        return games_df.sort_values('game_id')
        
    except Exception as e:
        print(f"Error in load_game_history: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

def get_recent_performance(team_name, games_df, current_game_id, n=5):
    """獲取球隊最近表現"""
    try:
        # 確認 DataFrame 結構
        #print("\nDataFrame 資訊:")
        #print("列名:", games_df.columns.tolist())
       # print("資料形狀:", games_df.shape)
        
        #print(f"\n正在查找 {team_name} 的比賽記錄")
        
        # 檢查 games_df 是否為空
        if games_df.empty:
            print("警告: DataFrame 是空的")
            return [0] * 7
            
        # 確保 'name' 列存在
        if 'name' not in games_df.columns:
           # print("錯誤: 找不到 'name' 列")
            #print("可用的列:", games_df.columns.tolist())
            return [0] * 7
            
        #print(f"當前所有可用的球隊名稱: {sorted(games_df['name'].unique())}")
        
        # 找出球隊比賽記錄
        team_games = games_df[games_df['name'] == team_name].copy()
        #print(f"找到 {len(team_games)} 場比賽記錄")
        
        team_games = team_games[team_games['game_id'] < current_game_id]
        recent_games = team_games.nlargest(n, 'game_id')
        
        if len(recent_games) == 0:
            print(f"警告: {team_name} 沒有找到任何有效的比賽記錄")
            return [0] * 7
            
        # 計算統計數據
        stats = [
            recent_games['points'].mean(),
            recent_games['reb'].mean(),
            recent_games['ast'].mean(),
            (recent_games['points'] > recent_games['opponent_points']).mean(),
            (recent_games['points'] - recent_games['opponent_points']).mean(),
            ((recent_games['two_m'] + recent_games['trey_m']) / 
             (recent_games['two'] + recent_games['trey'])).mean(),
            (recent_games['ft_m'] / recent_games['ft']).mean()
        ]
        
        return stats
        
    except Exception as e:
        print(f"Error in get_recent_performance: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return [0] * 7

def standardize_player_data(player_data, season=None):
    """標準化球員資料中的球隊名稱"""
    player_data['team'] = get_standardized_team_name(player_data['team'], season)
    return player_data

# =====================
# 4. 球員評分系統
# =====================
# 球員評分權重
weights = {
    'G': {
        'points': 1.0, 'All_goals_pct': 0.8, 'field_goals_two_pct': 0.6,
        'field_goals_three_pct': 1.0, 'free_throws_pct': 0.7, 'rebounds': 0.5,
        'assists': 1.2, 'steals': 1.0, 'blocks': 0.3, 'turnovers': -0.8, 'fouls': -0.5
    },
    'F': {
        'points': 1.0, 'All_goals_pct': 0.8, 'field_goals_two_pct': 0.7,
        'field_goals_three_pct': 0.6, 'free_throws_pct': 0.6, 'rebounds': 1.0,
        'assists': 0.6, 'steals': 0.7, 'blocks': 0.8, 'turnovers': -0.7, 'fouls': -0.6
    },
    'C': {
        'points': 0.8, 'All_goals_pct': 0.7, 'field_goals_two_pct': 0.8,
        'field_goals_three_pct': 0.3, 'free_throws_pct': 0.5, 'rebounds': 1.2,
        'assists': 0.4, 'steals': 0.4, 'blocks': 1.0, 'turnovers': -0.6, 'fouls': -0.7
    }
}

def calculate_player_score(player_data):
    """計算球員評分"""
    player_name = player_data['player']
    player_position = position_dict.get(player_name, 'G')
    player_score = 0
    for stat, weight in weights[player_position].items():
        try:
            player_score += weight * float(player_data[stat])
        except (ValueError, KeyError):
            continue
    return player_score

# =====================
# 5. 比賽預測相關函數
# =====================
def get_recent_performance(team_name, games_df, current_game_id=1000, n=5):
    games_df = load_game_history()
    """獲取球隊最近表現，最多取n場比賽"""
    team_name = get_standardized_team_name(team_name)
    #print(f"\n正在查找 {team_name} 的比賽記錄")
    
    # 找出球隊比賽記錄
    team_games = games_df[games_df['name'] == team_name].copy()
    #print(f"找到總共 {len(team_games)} 場比賽記錄")
    
    if len(team_games) == 0:
        print(f"警告: {team_name} 沒有任何比賽記錄")
        return [0] * 7
    
    # 篩選出在指定ID之前的比賽
    team_games = team_games[team_games['game_id'] < current_game_id]
    #print(f"符合日期條件的比賽數: {len(team_games)}")
    
    # 取得最近n場或所有可用的比賽（取較小值）
    available_games = min(n, len(team_games))
    recent_games = team_games.nlargest(available_games, 'game_id')
    
    #print(f"\n最近 {available_games} 場比賽記錄:")
    #print(recent_games[['game_id', 'points', 'opponent_points']])
    
    # 計算統計數據
    avg_points = recent_games['points'].mean()
    avg_rebounds = recent_games['reb'].mean()
    avg_assists = recent_games['ast'].mean()
    win_rate = (recent_games['points'] > recent_games['opponent_points']).mean()
    avg_point_diff = (recent_games['points'] - recent_games['opponent_points']).mean()
    
    # 計算投籃命中率
    total_fg_made = recent_games['two_m'].sum() + recent_games['trey_m'].sum()
    total_fg_attempts = recent_games['two'].sum() + recent_games['trey'].sum()
    fg_pct = total_fg_made / total_fg_attempts if total_fg_attempts > 0 else 0
    
    # 計算罰球命中率
    total_ft_made = recent_games['ft_m'].sum()
    total_ft_attempts = recent_games['ft'].sum()
    ft_pct = total_ft_made / total_ft_attempts if total_ft_attempts > 0 else 0

    result = [avg_points, avg_rebounds, avg_assists, win_rate, 
             avg_point_diff, fg_pct, ft_pct]
    # region print something
    # print(f"\n{team_name} 的最近 {available_games} 場比賽統計:")
    # print(f"平均得分: {avg_points:.1f}")
    # print(f"平均籃板: {avg_rebounds:.1f}")
    # print(f"平均助攻: {avg_assists:.1f}")
    # print(f"勝率: {win_rate:.2f}")
    # print(f"場均得分差: {avg_point_diff:.1f}")
    # print(f"投籃命中率: {fg_pct:.3f}")
    # print(f"罰球命中率: {ft_pct:.3f}")
    # endregion
    return result


def find_team(team_name, all_team_data):
    """尋找球隊資料"""
    exact_match = next((team for team in all_team_data if team['team_name'] == team_name), None)
    if exact_match:
        return exact_match
    all_names = [team['team_name'] for team in all_team_data]
    close_matches = get_close_matches(team_name, all_names, n=1, cutoff=0.6)
    if close_matches:
        return next(team for team in all_team_data if team['team_name'] == close_matches[0])
    return None

def predict_game_result(team1, team2, is_home_team,current_game_id=1000):
    """使用機器學習模型預測比賽結果，加入近期表現權重"""
    #print(f"\n預測比賽: {team1} vs {team2}")
    #print(f"主場球隊: {is_home_team}")
    
    team1_name = get_standardized_team_name(team1)
    team2_name = get_standardized_team_name(team2)
    
    #print(f"\n球隊名稱標準化後: {team1_name} vs {team2_name}")

    team1_stats = find_team(team1_name, all_team_data)
    team2_stats = find_team(team2_name, all_team_data)
    
    if not team1_stats or not team2_stats:
        return f"找不到球隊數據: {team1_name if not team1_stats else ''} {team2_name if not team2_stats else ''}"

    # 準備機器學習特徵
   # print(f"\n獲取 {team1_name} 的統計數據...")
    team1_basic = [float(team1_stats['wins']), float(team1_stats['losses']), 
                   team_avg_points.get(team1_name, 0)]
    
    #print(f"\n獲取 {team2_name} 的統計數據...")
    team2_basic = [float(team2_stats['wins']), float(team2_stats['losses']), 
                   team_avg_points.get(team2_name, 0)]

    team1_players = [score for _, score in team_top_players.get(team1_name, [])][:9]
    team2_players = [score for _, score in team_top_players.get(team2_name, [])][:9]
    
    team1_players.extend([0] * (9 - len(team1_players)))
    team2_players.extend([0] * (9 - len(team2_players)))

    #print(f"\n獲取 {team1_name} 的最近表現...")
    team1_recent = get_recent_performance(team1_name, current_game_id)
    
    #print(f"\n獲取 {team2_name} 的最近表現...")
    team2_recent = get_recent_performance(team2_name, current_game_id)

    # 準備預測特徵
    X_pred = np.array([
        team1_basic + team1_players + team1_recent,
        team2_basic + team2_players + team2_recent
    ])

    # 輸出特徵值
    # print("\n預測特徵:")
    # print(f"{team1_name} 特徵:")
    # print(f"基本數據: {team1_basic}")
    # print(f"球員評分: {team1_players}")
    # print(f"最近表現: {team1_recent}")
    
    # print(f"\n{team2_name} 特徵:")
    # print(f"基本數據: {team2_basic}")
    # print(f"球員評分: {team2_players}")
    # print(f"最近表現: {team2_recent}")
    
    # 機器學習預測
    win_probs = clf.predict_proba(X_pred)
    ml_team1_win_prob = win_probs[0][1]
    ml_team2_win_prob = win_probs[1][1]

    # 計算近期表現影響
    team1_recent_win_rate = team1_recent[3]  # 索引3是最近的勝率
    team2_recent_win_rate = team2_recent[3]
    
    # 計算近期得分差影響
    team1_point_diff = team1_recent[4]  # 索引4是得分差
    team2_point_diff = team2_recent[4]
    
    # 正規化得分差為-1到1之間
    max_point_diff = max(abs(team1_point_diff), abs(team2_point_diff))
    if max_point_diff != 0:
        team1_point_diff_norm = team1_point_diff / (2 * max_point_diff) + 0.5
        team2_point_diff_norm = team2_point_diff / (2 * max_point_diff) + 0.5
    else:
        team1_point_diff_norm = 0.5
        team2_point_diff_norm = 0.5

    # 結合所有因素
    # ML預測權重0.6，近期勝率權重0.25，得分差權重0.15
    team1_win_prob = (0.7 * ml_team1_win_prob + 
                     0.15 * team1_recent_win_rate +
                     0.15 * team1_point_diff_norm)
    
    team2_win_prob = (0.7 * ml_team2_win_prob + 
                     0.15 * team2_recent_win_rate +
                     0.15 * team2_point_diff_norm)

    # 主場優勢調整
    home_advantage = 1.1  # 10%的主場優勢
    if is_home_team == team1_name:
        team1_win_prob *= home_advantage
    elif is_home_team == team2_name:
        team2_win_prob *= home_advantage

    # 正規化勝率
    total_prob = team1_win_prob + team2_win_prob
    team1_win_prob = team1_win_prob / total_prob
    team2_win_prob = team2_win_prob / total_prob

    # 預測得分
    team1_score_pred = team1_recent[0] if team1_recent[0] > 0 else team_avg_points.get(team1_name, 85)
    team2_score_pred = team2_recent[0] if team2_recent[0] > 0 else team_avg_points.get(team2_name, 85)

    # 根據勝率調整得分
    score_adjustment = 5
    if team1_win_prob > team2_win_prob:
        team1_score_pred += score_adjustment * (team1_win_prob - 0.5)
        team2_score_pred -= score_adjustment * (team1_win_prob - 0.5)
    else:
        team1_score_pred -= score_adjustment * (team2_win_prob - 0.5)
        team2_score_pred += score_adjustment * (team2_win_prob - 0.5)

    # 主場得分調整
    if is_home_team == team1_name:
        team1_score_pred *= 1.02
    elif is_home_team == team2_name:
        team2_score_pred *= 1.02

    # print("\n預測詳情:")
    # print(f"機器學習預測 - {team1_name}勝率: {ml_team1_win_prob:.3f}, {team2_name}勝率: {ml_team2_win_prob:.3f}")
    # print(f"近期表現 - {team1_name}勝率: {team1_recent_win_rate:.3f}, {team2_name}勝率: {team2_recent_win_rate:.3f}")
    # print(f"得分差影響 - {team1_name}: {team1_point_diff_norm:.3f}, {team2_name}: {team2_point_diff_norm:.3f}")
    # print(f"最終預測 - {team1_name}勝率: {team1_win_prob:.3f}, {team2_name}勝率: {team2_win_prob:.3f}")

    return f"預測{team1_name}對{team2_name}的比賽結果:\n" \
           f"  {team1_name}勝率: {team1_win_prob:.2f}\n" \
           f"  {team2_name}勝率: {team2_win_prob:.2f}\n" \
           f"  {team1_name}預測得分: {team1_score_pred:.1f}\n" \
           f"  {team2_name}預測得分: {team2_score_pred:.1f}\n" \
           f"  主場隊伍: {is_home_team}"
if __name__ == "__main__":
    # 讀取數據
    games_df = load_game_history()
    
    # 讀取JSON數據
    p_player_data_23 = load_json('../dataset/P_Players_Performance_23_24.json')
    t1_player_data_23 = load_json('../dataset/T1_Players_performance_23_24.json')
    p_player_data_24 = load_json('../dataset/P_Players_Performance_24_25.json')
    t1_player_data_24 = load_json('../dataset/T1_Players_performance_24_25.json')

    p_team_data_23 = load_json('../dataset/P_TeamStanding23_24.json')
    t1_team_data_23 = load_json('../dataset/T1_TeamStanding23_24.json')
    p_team_data_24 = load_json('../dataset/P_TeamStanding24_25.json')
    t1_team_data_24 = load_json('../dataset/T1_TeamStanding24_25.json')

    t1_season_data = load_json('../dataset/T1_Season_teams_performance_24_25.json')
    p_season_data = load_json('../dataset/P_Season_teams_Performance_24_25.json')

    # 標準化球員數據
    all_player_data = []
    for player in (p_player_data_23 + t1_player_data_23):
        all_player_data.append(standardize_player_data(player, "23-24"))
    for player in (p_player_data_24 + t1_player_data_24):
        all_player_data.append(standardize_player_data(player, "24-25"))

    # 合併所有數據
    all_team_data = p_team_data_23 + t1_team_data_23 + p_team_data_24 + t1_team_data_24
    all_season_data = t1_season_data + p_season_data

    # 建立球員位置字典
    position_dict = {player_data['player']: player_data['position'] for player_data in all_player_data}

    # 計算每支球隊的最強9人
    team_top_players = defaultdict(list)
    for player_data in all_player_data:
        player_name = player_data['player']
        player_team = player_data['team']
        player_score = calculate_player_score(player_data)
        team_top_players[player_team].append((player_name, player_score))

    for team, players in team_top_players.items():
        team_top_players[team] = sorted(players, key=lambda x: x[1], reverse=True)[:9]

    # 創建球隊平均得分字典
    team_avg_points = {team['team']: float(team['points']) for team in all_season_data}

    # 準備訓練資料
    X = []
    y = []
    for team_data in all_team_data:
        team_name = get_standardized_team_name(team_data['team_name'])
        recent_perf = get_recent_performance(team_name, games_df, float('inf'))
        
        basic_features = [team_data['wins'], team_data['losses'], team_avg_points.get(team_name, 0)]
        player_features = [score for _, score in team_top_players.get(team_name, [])][:9]
        player_features.extend([0] * (9 - len(player_features)))
        
        X.append(basic_features + player_features + recent_perf)
        pct = float(team_data['pct'].rstrip('%')) / 100
        y.append(1 if pct > 0.5 else 0)

    X = np.array(X)
    y = np.array(y)
    
    # 訓練模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # 儲存訓練好的模型
    joblib.dump(clf, 'model.pkl')
    
    model_data = {
        'team_avg_points': team_avg_points,
        'team_top_players': dict(team_top_players),  # 轉換 defaultdict 為一般 dict
        'all_team_data': all_team_data
    }
    
    with open('model_data.json', 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=4)

    print("模型和數據已儲存")

# print(predict_game_result("臺南台鋼獵鷹", "高雄鋼鐵人", "臺南台鋼獵鷹"))