import streamlit as st
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ปิดการแจ้งเตือน Warning
warnings.filterwarnings('ignore')

# --- 1. การตั้งค่าหน้าเว็บ (Configuration) ---
st.set_page_config(page_title="Intelligent System 2568", layout="wide")

# --- Custom CSS เพื่อความสวยงามและแก้บัคสีตัวเลข ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
        font-weight: 800 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    [data-testid="stMetricLabel"] p {
        color: #333333 !important;
        font-size: 1.1rem !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        border: 1px solid #d1d1d1;
    }

    .main { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Sarabun', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ฟังก์ชันเตรียมข้อมูล (Preprocessing Function)
def clean_data(df):
    df = df.drop_duplicates()
    df.columns = [col.lower() for col in df.columns]
    df = df.replace(r"\?", np.nan, regex=True)
    for col in df.columns:
        if col not in ['origin', 'department', 'name', 'attrition']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ฟังก์ชันสร้างโครงสร้าง 1D-CNN สำหรับ Model 2
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        Conv1D(32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 2. ส่วนการประมวลผลโมเดล (Train & Evaluate) ---
@st.cache_resource
def train_and_get_metrics():
    # --- โมเดลที่ 1: Employee (Ensemble Learning) ---
    df_e = pd.read_csv("employee_data_clean.csv")
    df_e = clean_data(df_e)
    df_e = df_e[df_e['monthly_income'] > 0]
    df_e['income_per_age'] = df_e['monthly_income'] / (df_e['age'] + 1)
    
    X_e = df_e.drop(columns=['attrition', 'emp_id'], errors='ignore')
    y_e = df_e['attrition']
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_e, y_e, test_size=0.2, random_state=42)
    
    num_cols_e = X_e.select_dtypes(include=np.number).columns
    cat_cols_e = X_e.select_dtypes(include='object').columns
    
    pre_e = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())]), num_cols_e),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='unknown')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_cols_e)
    ])
    
    model_e = Pipeline([('pre', pre_e), ('clf', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))], voting='soft'))])
    
    model_e.fit(X_train_e, y_train_e)
    acc_e = accuracy_score(y_test_e, model_e.predict(X_test_e))

    # --- โมเดลที่ 2: Automobile (1D-CNN) ---
    df_a = pd.read_csv("Automobile_clean.csv")
    df_a = clean_data(df_a)
    X_a = df_a.drop(columns=['origin', 'name'], errors='ignore')
    
    # จัดการ Label
    le_a = LabelEncoder()
    y_a = le_a.fit_transform(df_a['origin'].fillna(df_a['origin'].mode()[0]))
    num_classes = len(le_a.classes_)
    
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    
    # Preprocessing สำหรับ CNN
    scaler_a = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    X_train_scaled = scaler_a.fit_transform(X_train_a)
    X_test_scaled = scaler_a.transform(X_test_a)
    
    # Reshape สำหรับ 1D-CNN (samples, features, 1)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # สร้างและเทรน CNN
    model_cnn = build_cnn_model(X_train_scaled.shape[1], num_classes)
    model_cnn.fit(X_train_cnn, y_train_a, epochs=30, batch_size=8, verbose=0)
    
    acc_a = model_cnn.evaluate(X_test_cnn, y_test_a, verbose=0)[1]
    
    return model_e, (model_cnn, scaler_a, le_a), acc_e, acc_a

# โหลดข้อมูลโมเดล
try:
    model_emp, model_auto_pack, score_emp, score_auto = train_and_get_metrics()
    model_cnn, auto_scaler, auto_le = model_auto_pack
except Exception as e:
    st.error(f"Error: กรุณาตรวจสอบไฟล์ CSV ในโฟลเดอร์: {e}")

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.title("Intelligent System 2568 Navigation")
    st.markdown("---")
    menu = st.radio("Main Menu", [
        "Employee Model Info", 
        "Automobile Model Info", 
        "Test System: Employee", 
        "Test System: Automobile"
    ])
    st.markdown("---")
    st.caption("Developed for Intelligent System Project 2026")

# --- หน้าที่ 1: รายละเอียด Employee (Machine Learning) ---
if menu == "Employee Model Info":
    st.title("Employee Attrition Analysis: Machine Learning Model")
    st.error("หมายเหตุเชิงวิชาการ: ชุดข้อมูลพนักงานนี้มีลักษณะเป็น Synthetic Data (Gen AI) ซึ่งถูกสร้างขึ้นเพื่อจำลองสถานการณ์การลาออก")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Advanced Data Preprocessing")
        st.markdown("""
        กระบวนการจัดเตรียมข้อมูล (Data Wrangling):
        * **Multi-Strategy Imputation**: เลือกใช้ Median Imputation เพื่อเติมเต็มข้อมูลเชิงปริมาณ
        * **Feature Synthesis**: สร้างฟีเจอร์ใหม่ Income per Age เพื่อวิเคราะห์นัยสำคัญของรายได้เทียบกับช่วงอายุ
        * **Robust Scaling Transformation**: ใช้ RobustScaler เพื่อจัดการกับข้อมูลที่มี Outliers สูง
        * **Categorical Handling**: ใช้ One-Hot Encoding แปลงชื่อแผนกให้เป็นรหัสไบนารี
        """)
        
        st.subheader("Algorithm Architecture: Hybrid Ensemble")
        st.markdown("""
        เราเลือกใช้แนวคิด Ensemble Learning:
        1. **Random Forest**: สร้างต้นไม้ตัดสินใจจำนวนมากและใช้การโหวตคะแนนข้างมาก
        2. **Gradient Boosting**: เรียนรู้จากความผิดพลาดของรอบก่อนหน้าเพื่อลดค่า Loss
        3. **Logistic Regression**: ทำหน้าที่เป็นโมเดลพื้นฐานในการประเมินความน่าจะเป็น
        """)

    with col2:
        st.subheader("Development & Evaluation Workflow")
        st.markdown("""
        ขั้นตอนการพัฒนา:
        1. **Data Acquisition & Cleansing**: การรวบรวมข้อมูลจำลองและกำจัด Noise
        2. **Automated Pipeline Construction**: ใช้ sklearn.pipeline เพื่อรวบรวมขั้นตอนทั้งหมดเข้าด้วยกัน
        3. **Train-Test Split Validation**: การแบ่งข้อมูล 80/20 เพื่อวัดประสิทธิภาพกับข้อมูลที่ไม่เคยเห็น
        """)
        
        st.subheader("Data Limitation Analysis")
        st.warning("""
        วิเคราะห์ข้อจำกัด:
        * **High Data Entropy**: ข้อมูลจำลองจาก AI อาจขาดตรรกะพฤติกรรมแฝงที่ซับซ้อนแบบมนุษย์จริง
        * **Non-Linear Complexity**: เส้นแบ่งเขตการตัดสินใจมีความซับซ้อนสูงเกินกว่าฟีเจอร์พื้นฐาน
        """)

# --- หน้าที่ 2: รายละเอียด Automobile (1D-CNN) ---
elif menu == "Automobile Model Info":
    st.title("Automobile Origin Classification: 1D-CNN Model")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Neural Network Architecture")
        st.markdown("""
        โมเดลนี้ถูกออกแบบโครงสร้างเอง (Custom Architecture) โดยใช้หลักการของ CNN:
        * **Conv1D Layer**: ทำหน้าที่สกัดความสัมพันธ์เชิงลึกระหว่างตัวแปรทางวิศวกรรม (เช่น แรงม้า, น้ำหนัก, ความเร็ว)
        * **Flatten Layer**: แปลงผลลัพธ์จาก Convolution ให้เป็นเวกเตอร์แบนราบ
        * **Dense Layer**: ชั้นประมวลผลเชิงลึกขนาด 64 นิวรอน พร้อม Dropout เพื่อป้องกัน Overfitting
        * **Softmax Output**: ทำหน้าที่พยากรณ์ความน่าจะเป็นของ 3 แหล่งกำเนิด (USA, Europe, Japan)
        """)
        
    with col2:
        st.subheader("Deep Learning Preprocessing")
        st.info("""
        กระบวนการเตรียมข้อมูลสำหรับ CNN:
        * **Standardization**: ปรับสเกลข้อมูลให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1
        * **Dimensional Reshaping**: ปรับรูปทรงข้อมูลจาก (Samples, Features) ให้เป็น (Samples, Features, 1) เพื่อให้สอดคล้องกับ Input ของ Convolutional Layer
        """)
        
        st.subheader("Dataset Source")
        st.markdown("ใช้ชุดข้อมูลจาก UCI Machine Learning Repository (Auto MPG Dataset) ซึ่งรวบรวมข้อมูลรถยนต์ช่วงปี 1970 - 1982")

# --- หน้าที่ 3: ทดสอบ Employee ---
elif menu == "Test System: Employee":
    st.title("Attrition Prediction Lab")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric(label="Model Accuracy", value=f"{score_emp*100:.2f} %", delta="Ensemble")
    
    tab1, tab2 = st.tabs(["Dataset Viewer", "Prediction Tool"])
    
    with tab1:
        df_emp = pd.read_csv("employee_data_clean.csv")
        st.dataframe(df_emp, use_container_width=True)

    with tab2:
        with st.form("emp_form"):
            st.subheader("Input Employee Parameters")
            c1, c2 = st.columns(2)
            age = c1.slider("Age", 18, 65, 30)
            income = c1.number_input("Monthly Income (THB)", value=40000)
            dept = c2.selectbox("Department", ["it", "marketing", "sales", "hr"])
            years = c2.number_input("Years at Company", value=3)
            score = st.select_slider("Performance Score", options=[1, 2, 3, 4, 5], value=3)
            
            submit = st.form_submit_button("Run Analysis")
            if submit:
                in_df = pd.DataFrame([[age, income, dept, years, score]], 
                                    columns=['age', 'monthly_income', 'department', 'years_at_company', 'performance_score'])
                in_df['income_per_age'] = in_df['monthly_income'] / (in_df['age'] + 1)
                res = model_emp.predict(in_df)
                
                if res[0] == 1:
                    st.error("Result: High Risk of Attrition")
                else:
                    st.success("Result: Likely to Stay")

# --- หน้าที่ 4: ทดสอบ Automobile ---
elif menu == "Test System: Automobile":
    st.title("Car Origin Classifier (1D-CNN)")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric(label="CNN Model Accuracy", value=f"{score_auto*100:.2f} %", delta="1D-CNN")
    with m2:
        df_a_plot = pd.read_csv("Automobile_clean.csv")
        st.bar_chart(df_a_plot['origin'].value_counts())

    st.markdown("### Technical Specifications Input")
    with st.form("auto_form"):
        c1, c2, c3 = st.columns(3)
        mpg = c1.number_input("MPG", value=20.0)
        cyl = c2.number_input("Cylinders", value=4.0)
        disp = c3.number_input("Displacement", value=150.0)
        hp = c1.number_input("Horsepower", value=100.0)
        weight = c2.number_input("Weight", value=3000.0)
        acc = c3.number_input("Acceleration", value=15.0)
        year = st.slider("Model Year (1970-1982)", 70, 82, 75)
        
        if st.form_submit_button("Identify Origin"):
            # เตรียมข้อมูล input ให้ตรงกับที่เทรน
            in_df = pd.DataFrame([[mpg, cyl, disp, hp, weight, acc, year]], 
                                columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year'])
            
            # 1. Scale ข้อมูล
            in_scaled = auto_scaler.transform(in_df)
            # 2. Reshape เป็น 3D สำหรับ CNN
            in_cnn = in_scaled.reshape(1, in_scaled.shape[1], 1)
            # 3. พยากรณ์
            res_prob = model_cnn.predict(in_cnn, verbose=0)
            res_class = np.argmax(res_prob, axis=1)
            res_label = auto_le.inverse_transform(res_class)
            
            st.success(f"Predicted Origin: {res_label[0].upper()}")
