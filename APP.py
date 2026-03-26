import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. การตั้งค่าหน้าเว็บ (Configuration) ---
st.set_page_config(page_title="Intelligent System 2568", layout="wide", page_icon="🤖")

# --- Custom CSS (บังคับสีตัวเลขให้ชัดเจน) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricText"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. กำหนดค่าเริ่มต้น (ป้องกัน NameError) ---
score_emp = 0.61  # ใส่ค่า Accuracy ที่ได้จากตอนเทรนในเครื่อง
score_auto = 0.85 # ใส่ค่า Accuracy ที่ได้จากตอนเทรนในเครื่อง
model_emp = None
model_auto = None

# --- 3. ฟังก์ชันโหลดโมเดล (โหลดจากไฟล์ .pkl) ---
@st.cache_resource
def load_trained_models():
    # ตรวจสอบชื่อไฟล์ใน GitHub ให้ตรงกับที่โหลด (ตัวเล็ก-ใหญ่มีผล)
    m_emp = joblib.load('employee_model.pkl')
    m_auto = joblib.load('automobile_model.pkl')
    return m_emp, m_auto

# พยายามโหลดโมเดล
try:
    model_emp, model_auto = load_trained_models()
except Exception as e:
    st.error(f"⚠️ ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
    st.info("ตรวจสอบว่ามีไฟล์ employee_model.pkl และ automobile_model.pkl บน GitHub หรือยัง")

# --- 4. แถบเมนูข้าง (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Project IS 2568")
    menu = st.radio("เมนูหลัก", [
        "🏠 หน้าแรก (Home)",
        "📘 ข้อมูลโมเดล Employee", 
        "🧪 ทดสอบ: Employee Attrition",
        "📗 ข้อมูลโมเดล Automobile", 
        "🧪 ทดสอบ: Automobile Origin"
    ])
    st.markdown("---")
    st.caption("สถานะโมเดล: พร้อมใช้งาน ✅" if model_emp else "สถานะโมเดล: ขัดข้อง ❌")

# --- 5. การแสดงผลเนื้อหาแต่ละหน้า ---

if menu == "🏠 หน้าแรก (Home)":
    st.title("🤖 ระบบวิเคราะห์ข้อมูลอัจฉริยะ")
    st.subheader("ยินดีต้อนรับเข้าสู่โครงงานวิชา Intelligent System")
    st.write("โปรเจคนี้ประกอบไปด้วยโมเดลทำนายการลาออกของพนักงาน และการจำแนกสัญชาติรถยนต์")

elif menu == "📘 ข้อมูลโมเดล Employee":
    st.title("👨‍💼 Model Information: Employee Attrition")
    st.markdown("""
    ### รายละเอียดโมเดล
    - **Algorithm:** Voting Classifier (Ensemble Learning)
    - **Base Models:** Random Forest, Gradient Boosting, Logistic Regression
    - **Preprocessing:** RobustScaler & OneHotEncoder
    """)

elif menu == "🧪 ทดสอบ: Employee Attrition":
    st.title("🧪 ระบบทำนายการลาออก")
    st.metric("Model Accuracy", f"{score_emp*100:.2f} %")
    
    if model_emp is not None:
        with st.form("emp_form"):
            c1, c2 = st.columns(2)
            age = c1.slider("อายุ (Age)", 18, 65, 30)
            income = c1.number_input("เงินเดือน (Monthly Income)", value=30000)
            dept = c2.selectbox("แผนก (Department)", ["it", "marketing", "sales", "hr"])
            years = c2.number_input("จำนวนปีที่ทำงาน", value=2)
            perf = st.select_slider("คะแนนประเมิน (Performance)", options=[1,2,3,4,5], value=3)
            
            if st.form_submit_button("วิเคราะห์ผล"):
                df_input = pd.DataFrame([[age, income, dept, years, perf]], 
                                       columns=['age', 'monthly_income', 'department', 'years_at_company', 'performance_score'])
                df_input['income_per_age'] = df_input['monthly_income'] / (df_input['age'] + 1)
                
                prediction = model_emp.predict(df_input)
                if prediction[0] == 1:
                    st.error("### ⚠️ ผลการทำนาย: มีความเสี่ยงในการลาออกสูง")
                else:
                    st.success("### ✅ ผลการทำนาย: พนักงานมีแนวโน้มทำงานต่อ")
    else:
        st.warning("ไม่สามารถทดสอบได้เนื่องจากโหลดโมเดลไม่สำเร็จ")

elif menu == "📗 ข้อมูลโมเดล Automobile":
    st.title("🚗 Model Information: Automobile Origin")
    st.markdown("""
    ### รายละเอียดโมเดล
    - **Algorithm:** Gradient Boosting Classifier
    - **Preprocessing:** SimpleImputer (Median) & StandardScaler
    - **Target:** จำแนกประเทศผู้ผลิต (USA, Europe, Japan)
    """)

elif menu == "🧪 ทดสอบ: Automobile Origin":
    st.title("🧪 ระบบจำแนกสัญชาติรถยนต์")
    st.metric("Model Accuracy", f"{score_auto*100:.2f} %")
    
    if model_auto is not None:
        with st.form("auto_form"):
            c1, c2, c3 = st.columns(3)
            mpg = c1.number_input("MPG", value=20.0)
            cyl = c2.number_input("Cylinders", value=4.0)
            disp = c3.number_input("Displacement", value=150.0)
            hp = c1.number_input("Horsepower", value=100.0)
            wt = c2.number_input("Weight", value=3000.0)
            acc = c3.number_input("Acceleration", value=15.0)
            yr = st.slider("Model Year (70-82)", 70, 82, 75)
            
            if st.form_submit_button("จำแนกสัญชาติ"):
                df_auto = pd.DataFrame([[mpg, cyl, disp, hp, wt, acc, yr]], 
                                      columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year'])
                res = model_auto.predict(df_auto)
                st.balloons()
                st.success(f"### 🌎 สัญชาติรถยนต์คือ: {res[0].upper()}")
