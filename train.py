import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ฟังก์ชันจัดการข้อมูลที่ไม่สะอาด (Data Cleaning) ตามโจทย์ข้อ 2
def clean_data(df):
    df = df.drop_duplicates()
    df.columns = [col.lower() for col in df.columns]
    # จัดการเครื่องหมาย ? ให้เป็นค่าว่าง
    df = df.replace(r"\?", np.nan, regex=True)
    # แปลงคอลัมน์ตัวเลขที่อาจถูกมองเป็น String ให้เป็น Numeric
    for col in df.columns:
        if col not in ['origin', 'department', 'name', 'attrition']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- ส่วนที่ 1: Train Employee Model (Ensemble) ---
df_e = pd.read_csv("employee_data_clean.csv")
df_e = clean_data(df_e)
df_e = df_e[df_e['monthly_income'] > 0] # ลบข้อมูลรายได้ที่ผิดปกติ
df_e['income_per_age'] = df_e['monthly_income'] / (df_e['age'] + 1) # Feature Engineering

X_e = df_e.drop(columns=['attrition', 'emp_id'], errors='ignore')
y_e = df_e['attrition']
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_e, y_e, test_size=0.2, random_state=42)

num_cols_e = X_e.select_dtypes(include=np.number).columns
cat_cols_e = X_e.select_dtypes(include='object').columns

pre_e = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())]), num_cols_e),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='unknown')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_cols_e)
])

# Ensemble Model: RF + GB + LR
model_e = Pipeline([('pre', pre_e), ('clf', VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))], voting='soft'))])
model_e.fit(X_train_e, y_train_e)
acc_e = accuracy_score(y_test_e, model_e.predict(X_test_e))

# --- ส่วนที่ 2: Train Automobile Model (GBC แทน Neural Network เพื่อความเสถียรบน Cloud) ---
df_a = pd.read_csv("Automobile_clean.csv")
df_a = clean_data(df_a)
X_a = df_a.drop(columns=['origin', 'name'], errors='ignore')
y_a = df_a['origin'].fillna(df_a['origin'].mode()[0])
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)

pre_a = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
model_a = Pipeline([('pre', pre_a), ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))])
model_a.fit(X_train_a, y_train_a)
acc_a = accuracy_score(y_test_a, model_a.predict(X_test_a))

# บันทึกทุกอย่างลงไฟล์เดียว
joblib.dump({
    'model_emp': model_e,
    'model_auto': model_a,
    'acc_emp': acc_e,
    'acc_auto': acc_a
}, 'trained_models.joblib')

print(f"✅ สำเร็จ! Employee Acc: {acc_e:.2f}, Auto Acc: {acc_a:.2f}")