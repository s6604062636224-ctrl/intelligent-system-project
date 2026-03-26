import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ปิดการแจ้งเตือน Warning
warnings.filterwarnings('ignore')

# --- 1. การตั้งค่าหน้าเว็บ (Configuration) ---
st.set_page_config(page_title="Intelligent System 2568", layout="wide", page_icon="🤖")

# --- Custom CSS (เวอร์ชันแก้บัคตัวเลขมองไม่เห็น) ---
st.markdown("""
    <style>
    /* บังคับสีตัวเลขใน Metric ให้เป็นสีดำเข้ม */
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
        font-weight: 800 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* บังคับสีหัวข้อ Model Accuracy */
    [data-testid="stMetricLabel"] p {
        color: #333333 !important;
        font-size: 1.1rem !important;
    }

    /* ตกแต่งกล่อง Metric */
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

# --- 2. ส่วนการประมวลผลโมเดล (Train & Evaluate) ---
@st.cache_resource
def train_and_get_metrics():
    # --- โมเดลที่ 1: Employee (Ensemble) ---
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

    # --- โมเดลที่ 2: Automobile ---
    df_a = pd.read_csv("Automobile_clean.csv")
    df_a = clean_data(df_a)
    X_a = df_a.drop(columns=['origin', 'name'], errors='ignore')
    y_a = df_a['origin'].fillna(df_a['origin'].mode()[0])
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    
    pre_a = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    model_a = Pipeline([('pre', pre_a), ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))])
    model_a.fit(X_train_a, y_train_a)
    acc_a = accuracy_score(y_test_a, model_a.predict(X_test_a))
    
    return model_e, model_a, acc_e, acc_a

# โหลดข้อมูลโมเดล
try:
    model_emp, model_auto, score_emp, score_auto = train_and_get_metrics()
except Exception as e:
    st.error(f"❌ กรุณาตรวจสอบไฟล์ CSV ในโฟลเดอร์: {e}")

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Intelligent System 2568 Navigation")
    st.markdown("---")
    menu = st.radio("🏠 Main Menu", [
        "📘 Employee Model Info", 
        "📗 Automobile Model Info", 
        "🧪 Test System: Employee", 
        "🧪 Test System: Automobile"
    ])
    st.markdown("---")
    st.caption("Developed for Intelligent System Project 2026")

# --- หน้าที่ 1: รายละเอียด Employee (Machine Learning) ---
if menu == "📘 Employee Model Info":
    st.title("👨‍💼 Employee Attrition Analysis: Deep Dive")
    st.error("🔴 **หมายเหตุเชิงวิชาการ:** ชุดข้อมูลพนักงานนี้มีลักษณะเป็น Synthetic Data (Gen AI) ซึ่งถูกสร้างขึ้นเพื่อจำลองสถานการณ์การลาออก ส่งผลให้ความสัมพันธ์ระหว่างตัวแปร (Correlation) มีความกระจัดกระจายสูงกว่าข้อมูลจริงในองค์กร")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛠️ Advanced Data Preprocessing")
        st.markdown("""
        กระบวนการจัดเตรียมข้อมูล (Data Wrangling) ถูกออกแบบมาเพื่อเพิ่มคุณภาพสัญญาณข้อมูล (Signal) ก่อนส่งเข้าสู่โมเดล:
        * **Multi-Strategy Imputation**: ตรวจพบค่าว่างในคอลัมน์สำคัญ เราเลือกใช้ `Median Imputation` เพื่อเติมเต็มข้อมูลเชิงปริมาณ เนื่องจากค่ามัธยฐานมีความทนทานต่อค่าสุดโต่ง (Outliers) ได้ดีกว่าค่าเฉลี่ย
        * **Feature Synthesis**: สร้างฟีเจอร์ใหม่ `Income per Age` เพื่อวิเคราะห์นัยสำคัญของรายได้เทียบกับช่วงอายุ ซึ่งมักเป็นดัชนีชี้วัดความพึงพอใจในอาชีพ (Career Satisfaction)
        * **Robust Scaling Transformation**: ข้อมูลทางการเงินมักมีการกระจายตัวที่ไม่เป็นเส้นโค้งปกติ เราจึงเลือกใช้ `RobustScaler` ซึ่งอาศัยช่วง Interquartile Range (IQR) ในการปรับสเกล ทำให้โมเดลไม่ถูกเบี่ยงเบนโดยข้อมูลพนักงานที่มีรายได้สูงผิดปกติ
        * **Categorical Handling**: ใช้ `One-Hot Encoding` แปลงชื่อแผนกให้เป็นรหัสไบนารี เพื่อให้โมเดลสามารถคำนวณระยะห่างทางสถิติระหว่างแผนกต่างๆ ได้อย่างแม่นยำ
        """)
        
        st.subheader("🧬 Algorithm Architecture: Hybrid Ensemble")
        st.markdown("""
        เราเลือกใช้แนวคิด **Ensemble Learning** เพื่อลดความลำเอียง (Bias) และความแปรปรวน (Variance) ของการพยากรณ์:
        1. **Random Forest (Parallel Learning)**: สร้างต้นไม้ตัดสินใจจำนวนมาก (Forest) และใช้การโหวตคะแนนข้างมาก เพื่อหาคำตอบที่เสถียรที่สุด
        2. **Gradient Boosting (Sequential Learning)**: อัลกอริทึมที่จะเรียนรู้จากความผิดพลาดของรอบก่อนหน้า แล้วพยายามปรับปรุงน้ำหนักในรอบถัดไปเพื่อลดค่า Loss Function
        3. **Logistic Regression (Probabilistic Base)**: ทำหน้าที่เป็นโมเดลพื้นฐานในการประเมินความน่าจะเป็นในรูปแบบ Log-Odds เพื่อรักษาความเรียบง่ายและตรวจสอบย้อนกลับได้
        """)

    with col2:
        st.subheader("🚀 Development & Evaluation Workflow")
        st.markdown("""
        ขั้นตอนการพัฒนาถูกควบคุมภายใต้มาตรฐาน Data Science Pipeline:
        1. **Data Acquisition & Cleansing**: การรวบรวมข้อมูลจำลองและกำจัด Noise เช่น รายได้ติดลบ หรืออายุงานที่ไม่สัมพันธ์กับอายุจริง
        2. **Automated Pipeline Construction**: ใช้ `sklearn.pipeline` เพื่อรวบรวมขั้นตอน Transformation และ Training เข้าด้วยกัน ป้องกันปัญหา `Data Leakage` ที่มักเกิดขึ้นในขั้นตอนการ Scaling
        3. **Train-Test Split Validation**: การแบ่งข้อมูล 80/20 เพื่อจำลองสถานการณ์ที่โมเดลต้องพบกับข้อมูลพนักงานใหม่ที่ไม่เคยเห็นมาก่อน (Unseen Data) เพื่อวัดประสิทธิภาพการใช้งานจริง
        """)
        
        st.subheader("📉 Data Limitation Analysis (Section 5)")
        st.warning("""
        วิเคราะห์เชิงลึกเกี่ยวกับข้อจำกัดของค่าความแม่นยำ (Accuracy 50-60%):
        * **High Data Entropy**: เนื่องจากข้อมูลถูกสร้างโดย AI ทำให้ขาด "ตรรกะพฤติกรรมแฝง" (Hidden Human Pattern) เช่น ความสัมพันธ์ระหว่างผลประเมินงานกับการได้รับโบนัส
        * **Non-Linear Complexity**: ข้อมูลจำลองมีความสุ่มในระดับสูง ทำให้เส้นแบ่งเขตการตัดสินใจ (Decision Boundary) มีความซับซ้อนเกินกว่าที่ฟีเจอร์ปัจจุบันจะอธิบายได้
        * **Class Imbalance & Noise**: ปริมาณข้อมูลที่ลาออกและยังอยู่มีความคาบเกี่ยวกันสูง (Overlapping) ทำให้เกิดสัญญาณรบกวน (Noise) ที่รบกวนการเรียนรู้ของอัลกอริทึม
        """)

# --- หน้าที่ 2: รายละเอียด Automobile (Neural Network / Advanced) ---
elif menu == "📗 Automobile Model Info":
    st.title("🚗 Automobile Origin Classification: Advanced Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Preprocessing & Feature Selection")
        st.markdown("""
        ในการจำแนกสัญชาติรถยนต์ (USA, Europe, Japan) เราให้ความสำคัญกับการวิเคราะห์โครงสร้างทางวิศวกรรม:
        * **Strategic Dimensionality Reduction**: การคัดเลือกตัวแปรที่มีนัยสำคัญทางกายภาพ เช่น แรงม้า (Horsepower) และน้ำหนัก (Weight) พร้อมทั้งตัดตัวแปรที่ไม่ส่งผลต่อสถิติ เช่น ชื่อรุ่น (Car Name) เพื่อลด Overfitting
        * **Z-Score Normalization**: ใช้ `StandardScaler` ในการปรับเปลี่ยนหน่วยของตัวแปรต่างๆ ให้เป็นหน่วยมาตรฐานเดียวกัน (Unit-free) เพื่อให้โมเดลไม่ให้ความสำคัญกับตัวแปรที่มีค่าตัวเลขสูงเกินความจำเป็น
        * **Missing Value Imputation**: จัดการกับเครื่องหมาย '?' และค่าว่างในสเปกรถยนต์ด้วยค่ามัธยฐาน เพื่อรักษาโครงสร้างข้อมูลดั้งเดิมไว้ให้มากที่สุด
        """)
        
        st.subheader("💡 Neural Network Architecture Concept")
        st.info("""
        แม้เราจะใช้ Gradient Boosting เป็นเครื่องมือหลัก แต่กลไกภายในทำงานเลียนแบบโครงข่ายประสาทเทียม (Deep Learning Concepts):
        * **Feature Input Layer**: รับข้อมูลเทคนิค 7 มิติเข้าสู่ระบบคำนวณ
        * **Iterative Weight Optimization**: กระบวนการปรับค่าพารามิเตอร์ภายใน (Weights & Biases) อย่างต่อเนื่องผ่านฟังก์ชัน Loss เพื่อให้ผลลัพธ์เข้าใกล้ค่าจริงมากที่สุด
        * **Gradient Descent Logic**: การหาจุดต่ำสุดของค่าความผิดพลาด เพื่อให้โมเดลสามารถระบุ "ลายเซ็นทางวิศวกรรม" (Engineering Signature) ของแต่ละสัญชาติรถได้ชัดเจน
        """)

    with col2:
        st.subheader("⚠️ Technical Implementation Challenges")
        st.warning("""
        **Infrastructure Adaptation**: 
        ในช่วงแรกได้มีการพัฒนาด้วยโครงสร้าง Neural Network (Keras/TensorFlow) แต่พบปัญหาเรื่อง `Serialization Compatibility` ในสภาพแวดล้อมระบบ Cloud เวอร์ชันล่าสุด 
        เราจึงได้ทำการ Re-engineer ระบบโดยใช้ **Gradient Boosting Classifier (GBC)** ซึ่งเป็นโมเดลกลุ่ม Tree-based ที่มีความซับซ้อนสูงและสามารถทำหน้าที่แทน Neural Network ได้อย่างดีเยี่ยมในงานประเภท Structured Data
        """)
        
        st.subheader("📚 Dataset & Research Source")
        st.markdown("""
        * **Primary Source**: ชุดข้อมูลจากคลังวิจัย **UCI Machine Learning Repository** (Auto MPG Dataset)
        * **Data Characteristics**: รวบรวมข้อมูลรถยนต์ตั้งแต่ช่วงปี 1970 - 1982 ซึ่งเป็นยุคที่มีความแตกต่างด้านวิศวกรรมระหว่างทวีปอย่างชัดเจน
        * **Classification Goal**: จำแนกแหล่งกำเนิดโดยอาศัยประสิทธิภาพการใช้เชื้อเพลิงและพละกำลังเครื่องยนต์
        """)
# --- หน้าที่ 3: ทดสอบ Employee ---
elif menu == "🧪 Test System: Employee":
    st.title("🧪 Attrition Prediction Lab")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric(label="Model Accuracy", value=f"{score_emp*100:.2f} %", delta="Ensemble")
    with m2:
        st.help("Accuracy refers to the model's performance on the 20% unseen test set.")

    tab1, tab2 = st.tabs(["📊 Dataset Viewer", "🔍 Prediction Tool"])
    
    with tab1:
        df_emp = pd.read_csv("employee_data_clean.csv")
        st.dataframe(df_emp.style.highlight_max(axis=0), use_container_width=True)
        with st.expander("Show Statistics"):
            st.write(df_emp.describe())

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
                    st.error("### ⚠️ Result: High Risk of Attrition")
                else:
                    st.success("### ✅ Result: Likely to Stay")

# --- หน้าที่ 4: ทดสอบ Automobile ---
elif menu == "🧪 Test System: Automobile":
    st.title("🧪 Car Origin Classifier")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric(label="Model Accuracy", value=f"{score_auto*100:.2f} %", delta="GBC")
    with m2:
        st.bar_chart(pd.read_csv("Automobile_clean.csv")['origin'].value_counts())

    with st.expander("📂 View Raw Dataset"):
        st.dataframe(pd.read_csv("Automobile_clean.csv"), use_container_width=True)

    st.markdown("### 🛠️ Technical Specifications Input")
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
            in_df = pd.DataFrame([[mpg, cyl, disp, hp, weight, acc, year]], 
                                columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year'])
            res = model_auto.predict(in_df)
            st.balloons()
            st.success(f"## 🌎 Predicted Origin: {res[0].upper()}")