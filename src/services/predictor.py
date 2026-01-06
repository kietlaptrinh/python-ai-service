from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from src.config.settings import MODEL_PATH, THRESHOLD_WEAK_SUBJECT
from src.models.ml_models import get_model
import xgboost as xgb


def generate_detailed_suggestions(student_subject_df: pd.DataFrame, subject_name: str) -> str:
    if student_subject_df.empty:
        return f"Với môn {subject_name}, cần có thêm dữ liệu để đưa ra gợi ý chi tiết."

    # Lấy dữ liệu của bài thi gần nhất trong môn học này
    latest_performance = student_subject_df.iloc[-1]

    # Quy tắc 1: Tỉ lệ bỏ qua câu hỏi cao -> Thiếu tự tin hoặc quản lý thời gian kém
    if latest_performance['skipped_ratio'] > 0.4:
        return (
            f"🎯 **Môn {subject_name}:** Tỉ lệ bỏ qua câu hỏi của bạn khá cao. "
            f"Hãy thử làm quen với các dạng bài và luyện tập quản lý thời gian để có thể hoàn thành tất cả các câu hỏi."
        )

    # Quy tắc 2: Tỉ lệ làm sai cao -> Nắm chưa vững kiến thức
    if latest_performance['wrong_ratio'] > 0.5:
        return (
            f"🎯 **Môn {subject_name}:** Có vẻ bạn nắm chưa vững kiến thức nền tảng vì tỉ lệ làm sai còn cao. "
            f"Hãy tập trung ôn lại lý thuyết và làm thêm bài tập cơ bản."
        )

    # Quy tắc 3: Phong độ giảm sút so với lịch sử
    if latest_performance['avg_score_history'] > 0 and latest_performance['score'] < latest_performance['avg_score_history']:
         return (
            f"🎯 **Môn {subject_name}:** Phong độ gần đây của bạn có vẻ đi xuống so với trước đây. "
            f"Hãy xem lại các lỗi sai ở bài thi gần nhất để rút kinh nghiệm nhé."
        )
    
    # Quy tắc mặc định
    return (
        f"🎯 **Môn {subject_name}:** Bạn cần tiếp tục nỗ lực để cải thiện điểm số ở môn này."
    )

def train_model(features_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, Any]:
 
    models = {}
    metrics = {}
    feature_cols = ['time_taken', 'correct_ratio', 'wrong_ratio', 'skipped_ratio', 
        'total_questions', 'avg_score_history', 'last_score', 'exams_taken_count'
    ]

    for subject_id in features_df['subject_id'].unique():
        subject_features = features_df[features_df['subject_id'] == subject_id][feature_cols]
        subject_target = target_df[target_df['subject_id'] == subject_id]['score']

        if len(subject_features) < 2:
            continue  # Skip if not enough data

        X_train, X_test, y_train, y_test = train_test_split(
            subject_features, subject_target, test_size=0.2, random_state=42
        )

        model = get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse = None if np.isnan(rmse) else rmse
        mae = None if np.isnan(mae) else mae
        r2 = None if np.isnan(r2) else r2

        models[subject_id] = model
        metrics[subject_id] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    # Save models
    joblib.dump(models, MODEL_PATH)
    return metrics

def predict_scores(features_df: pd.DataFrame, subjects_dict: Dict[str, str]) -> Dict[str, Any]:
 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first.")

    models = joblib.load(MODEL_PATH)
    predictions = {}
    feature_cols = [
        'time_taken', 'correct_ratio', 'wrong_ratio', 'skipped_ratio',
        'total_questions', 'avg_score_history', 'last_score', 'exams_taken_count'
    ]

    for student_id in features_df['student_id'].unique():
        student_data = features_df[features_df['student_id'] == student_id]
        student_preds = {}
        for subject_id in student_data['subject_id'].unique():
            if subject_id in models:
                subject_features = student_data[student_data['subject_id'] == subject_id][feature_cols]
                if not subject_features.empty:
                    pred_score = models[subject_id].predict(subject_features)[0]
                    student_preds[subjects_dict.get(subject_id, subject_id)] = float(pred_score)
        predictions[student_id] = student_preds

    weak_subjects = {}
    ranks = {}
    suggestions = {}
    subject_name_to_id = {v: k for k, v in subjects_dict.items()}

    for student_id, preds in predictions.items():
        weak_subject_names = [subj for subj, score in preds.items() if score < THRESHOLD_WEAK_SUBJECT]
        weak_subjects[student_id] = weak_subject_names

        if not preds: # Nếu không có dự đoán nào (dictionary rỗng)
            suggestions[student_id] = "Chưa đủ dữ liệu lịch sử để dự đoán sức học. Hãy làm thêm bài tập nhé!"
        elif not weak_subject_names: # Có dự đoán và không có môn yếu
            suggestions[student_id] = "Chúc mừng! Bạn đang có phong độ rất tốt ở tất cả các môn. Hãy tiếp tục phát huy!"
        else:
            detailed_suggestions = []
            student_df = features_df[features_df['student_id'] == student_id]
            for name in weak_subject_names:
                subject_id = subject_name_to_id.get(name)
                if subject_id:
                    student_subject_df = student_df[student_df['subject_id'] == subject_id].sort_values(by='exam_id')
                    suggestion = generate_detailed_suggestions(student_subject_df, name)
                    detailed_suggestions.append(suggestion)
            suggestions[student_id] = "\n".join(detailed_suggestions)

        avg_score = np.mean(list(preds.values())) if preds else 0
        correct_ratio = features_df[features_df['student_id'] == student_id]['correct_ratio'].mean()
        if avg_score >= 9.0 and correct_ratio >= 0.9:
            rank = 'A'
        elif avg_score >= 8.0 and correct_ratio >= 0.8:
            rank = 'B'
        elif avg_score >= 7.0 and correct_ratio >= 0.7:
            rank = 'C'
        elif avg_score >= 5.0 and correct_ratio >= 0.5:
            rank = 'D'
        else:
            rank = 'F'
        ranks[student_id] = rank

  

    return {
        'predictions': predictions,
        'weak_subjects': weak_subjects,
        'ranks': ranks,
        'suggestions': suggestions
    }