from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
def get_model():
    return xgb.XGBRegressor(
        n_estimators=100,      # Số lượng cây
        learning_rate=0.1,     # Tốc độ học
        max_depth=5,           # Độ sâu tối đa của cây
        objective='reg:squarederror', # Mục tiêu là dự đoán số (hồi quy)
        random_state=42
    )