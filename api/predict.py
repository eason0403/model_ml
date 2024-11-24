import joblib
import json
from collections import defaultdict
import numpy as np
from model import get_standardized_team_name, find_team, get_recent_performance

# 載入模型和數據
clf = joblib.load('model.pkl')

# 載入必要的數據
with open('model_data.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)
    
team_avg_points = model_data['team_avg_points']
team_top_players = defaultdict(list, model_data['team_top_players'])
all_team_data = model_data['all_team_data']

# 這裡複製您原本的預測相關函數
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

# 使用範例
if __name__ == "__main__":
    result = predict_game_result("臺南台鋼獵鷹", "高雄鋼鐵人", "臺南台鋼獵鷹")
    print(result)