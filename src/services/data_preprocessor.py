from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.config.settings import SCALER_PATH

def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng dựa trên lịch sử của học sinh."""
    # Sắp xếp dữ liệu theo học sinh và thời gian (giả sử exam_id tăng dần theo thời gian)
    df = df.sort_values(by=['student_id', 'exam_id'])

    # Tính toán các đặc trưng lịch sử cho mỗi học sinh
    # Dùng groupby('student_id') để tính toán riêng cho từng người
    df['avg_score_history'] = df.groupby('student_id')['score'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['last_score'] = df.groupby('student_id')['score'].transform(lambda x: x.shift(1))
    df['exams_taken_count'] = df.groupby('student_id').cumcount()

    # Xử lý các giá trị NaN phát sinh ở bài thi đầu tiên
    df.fillna(0, inplace=True)
    
    return df

def preprocess_for_analysis(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    results_df = pd.DataFrame(data['results'])
    exams_df = pd.DataFrame(data['exams'])
    subjects_dict = data['subjects']

    # Handle missing values
    results_df['score'] = results_df['score'].fillna(0)
    results_df['time_taken'] = results_df['time_taken'].fillna(results_df['time_taken'].mean())

    # Create features
    feature_list = []
    for _, result in results_df.iterrows():
        exam_matches = exams_df[exams_df['id'].astype(str) == str(result['exam_id'])]
        if exam_matches.empty:
            continue
        exam = exam_matches.iloc[0]

        # 1. Chuẩn hóa ID câu hỏi sang String để khớp với JSON
        questions = {str(q['id']): str(q['correct_answer']).strip() for q in exam['questions']}
        
        # 2. Chuẩn hóa câu trả lời của học sinh (đưa key về string)
        student_answers = {str(k): str(v).strip() for k, v in result['answers'].items()}

        correct = wrong = skipped = 0
        for q_id, correct_ans in questions.items():
            if q_id in result['answers']:
                user_ans = student_answers[q_id]
                if user_ans.lower() == correct_ans.lower():
                    correct += 1
                else:
                    wrong += 1
            else:
                skipped += 1
        total = correct + wrong + skipped
        feature_list.append({
            'student_id': result['student_id'],
            'exam_id': result['exam_id'],
            'subject_id': exam['subject_id'],
            'score': result['score'],
            'time_taken': result['time_taken'],
            'correct': correct,
            'wrong': wrong,
            'skipped': skipped,
            'correct_ratio': correct / total if total > 0 else 0,
            'wrong_ratio': wrong / total if total > 0 else 0,
            'skipped_ratio': skipped / total if total > 0 else 0,
            'total_questions': total
        })

    features_df = pd.DataFrame(feature_list)

    # Thêm các đặc trưng lịch sử nhưng KHÔNG chuẩn hóa
    features_df = create_historical_features(features_df)
    
    # Trả về dữ liệu thô, đã được làm giàu
    return features_df, exams_df, subjects_dict

def preprocess_data(data: Dict[str, Any], is_training: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, str]]:
    """
    Preprocess input data: standardize, handle missing values, and create ML features.

    :param data: Input dictionary with results, exams, and subjects.
    :param is_training: Flag to indicate if preprocessing is for training (includes target).
    :return: Tuple of (features_df, target_df (if training), exams_df, subjects_dict).
    """
    results_df = pd.DataFrame(data['results'])
    exams_df = pd.DataFrame(data['exams'])
    subjects_dict = data['subjects']

    # Handle missing values
    results_df['score'] = results_df['score'].fillna(0)
    results_df['time_taken'] = results_df['time_taken'].fillna(results_df['time_taken'].mean())

    # Create features
    feature_list = []
    for _, result in results_df.iterrows():
        exam = exams_df[exams_df['id'] == result['exam_id']].iloc[0]
        questions = {q['id']: q['correct_answer'] for q in exam['questions']}
        correct = wrong = skipped = 0
        for q_id, correct_ans in questions.items():
            if q_id in result['answers']:
                user_ans = result['answers'][q_id]
                if user_ans == correct_ans:
                    correct += 1
                else:
                    wrong += 1
            else:
                skipped += 1
        total = correct + wrong + skipped
        feature_list.append({
            'student_id': result['student_id'],
            'exam_id': result['exam_id'],
            'subject_id': exam['subject_id'],
            'score': result['score'],
            'time_taken': result['time_taken'],
            'correct': correct,        
            'wrong': wrong,             
            'skipped': skipped,
            'correct_ratio': correct / total if total > 0 else 0,
            'wrong_ratio': wrong / total if total > 0 else 0,
            'skipped_ratio': skipped / total if total > 0 else 0,
            'total_questions': total
        })

    features_df = pd.DataFrame(feature_list)

    features_df = create_historical_features(features_df)

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = ['score', 'time_taken', 'correct_ratio', 'wrong_ratio', 
        'skipped_ratio', 'total_questions', 'avg_score_history', 
        'last_score', 'exams_taken_count']
    if is_training:
        features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])
        joblib.dump(scaler, SCALER_PATH)
    else:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            features_df[numerical_cols] = scaler.transform(features_df[numerical_cols])
        else:
            raise FileNotFoundError("Scaler not found. Train the model first.")

    # Prepare target for training
    target_df = features_df[['student_id', 'subject_id', 'score']] if is_training else None

    return features_df, target_df, exams_df, subjects_dict