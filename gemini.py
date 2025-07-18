import google.generativeai as genai
import streamlit as st

# ✅ Configure Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ✅ Use a lighter model to reduce quota issues
model_gemini = genai.GenerativeModel("models/gemini-pro")  # Avoid gemini-1.5-pro if hitting quota

# ✅ Cache Gemini responses (avoid repeated requests on same input)
@st.cache_data(show_spinner=False)
def generate_gemini_response(student_data, student_id=None):
    id_str = f" for Student ID {student_id}" if student_id else ""

    prompt = f"""
You're an educational advisor AI. Based on the following student data{id_str}, give supportive, practical advice in plain language.
Help the student understand their situation and suggest realistic ways to avoid dropping out and improve performance.

Student Info:
- Age: {student_data['Age']}
- CGPA: {student_data['CGPA']} / 5.0
- Attendance Rate: {student_data['Attendance Rate']}%
- Behavioural Rating: {student_data['Behavioural Rating']}%
- Study Time: {student_data['Study Time']} hrs/week
- Parental Support: {"Yes" if student_data['Parental Support'] == 1 else "No"}
- Extra Paid Class: {"Yes" if student_data['Extra Paid Class'] == 1 else "No"}

Respond with 3-5 sentences. Be encouraging and practical.
"""

    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"
