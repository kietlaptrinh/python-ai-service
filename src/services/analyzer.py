from typing import Dict, Any
import pandas as pd
from src.config.settings import THRESHOLD_WEAK_SUBJECT

def analyze_data(features_df: pd.DataFrame, exams_df: pd.DataFrame, subjects_dict: Dict[str, str]) -> Dict[str, Any]:

    # Map subject names
    exams_df['subject'] = exams_df['subject_id'].map(subjects_dict)

    # Average score per student
    avg_score_per_student = features_df.groupby('student_id')['score'].mean().to_dict()

    # Average score per subject
    avg_score_per_subject = features_df.groupby('subject_id')['score'].mean().to_dict()
    avg_score_per_subject = {subjects_dict[k]: v for k, v in avg_score_per_subject.items()}

    # Average time taken
    avg_time_taken = features_df['time_taken'].mean()

    # Weak subjects per student
    weak_subjects = {}
    for student, group in features_df.groupby('student_id'):
        weak = [subjects_dict[sub_id] for sub_id in group[group['score'] < THRESHOLD_WEAK_SUBJECT]['subject_id'].unique()]
        weak_subjects[student] = weak

    # Stats per student
    stats_per_student = features_df.groupby('student_id')[['correct', 'wrong', 'skipped']].sum()
    stats_per_student['total'] = stats_per_student.sum(axis=1)
    stats_per_student['correct_ratio'] = (stats_per_student['correct'] / stats_per_student['total']).fillna(0)
    stats_per_student['wrong_ratio'] = (stats_per_student['wrong'] / stats_per_student['total']).fillna(0)
    stats_per_student['skipped_ratio'] = (stats_per_student['skipped'] / stats_per_student['total']).fillna(0)
    stats_per_student = stats_per_student.to_dict('index')

    # Stats per subject
    stats_per_subject = features_df.groupby('subject_id')[['correct', 'wrong', 'skipped']].sum()
    stats_per_subject['total'] = stats_per_subject.sum(axis=1)
    stats_per_subject['correct_ratio'] = (stats_per_subject['correct'] / stats_per_subject['total']).fillna(0)
    stats_per_subject['wrong_ratio'] = (stats_per_subject['wrong'] / stats_per_subject['total']).fillna(0)
    stats_per_subject['skipped_ratio'] = (stats_per_subject['skipped'] / stats_per_subject['total']).fillna(0)
    stats_per_subject = {subjects_dict[k]: v for k, v in stats_per_subject.to_dict('index').items()}

    return {
        'avg_score_per_student': avg_score_per_student,
        'avg_score_per_subject': avg_score_per_subject,
        'avg_time_taken': avg_time_taken,
        'weak_subjects_per_student': weak_subjects,
        'stats_per_student': stats_per_student,
        'stats_per_subject': stats_per_subject
    }