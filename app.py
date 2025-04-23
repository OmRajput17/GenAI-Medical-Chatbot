import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import json
import base64
import io
from fpdf import FPDF
import re  # For extracting conditions from the AI response

# Import doctors database
from doctors_data import find_doctors_for_conditions

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Medical Symptom Chatbot With Ollama"

# Initialize session state for chat history and assessment
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""You are a highly knowledgeable and responsible AI medical assistant. Your task is to help users identify possible diseases based on the symptoms they describe.

        Instructions:

        1. Ask clear follow-up questions if needed.
        2. Predict the most likely diseases based on the symptoms.
        3. Provide concise reasoning for each condition you suggest.
        4. Be accurate, factual, and avoid guessing or inventing conditions.
        5. Always include this line at the end: "This is not a diagnosis. Please consult a doctor for medical advice."
        6. Only suggest well-known medical conditions based on symptoms provided.
        7. Also ask cross questions for clarification if needed.
        8. Also suggest some general medicines which are present which will not cause and side effects and also provide precautions and general tips.
        9. Also suggest dosctors from the dataset based on the symptoms provided.
        10. Be polite and professional in your responses.""")
    ]

if "enough_info" not in st.session_state:
    st.session_state.enough_info = False
    
if "final_assessment" not in st.session_state:
    st.session_state.final_assessment = None

if "recommended_doctors" not in st.session_state:
    st.session_state.recommended_doctors = {}

# Title of the app
st.title("Medical Symptom Analyzer")
st.subheader("Describe your symptoms to get potential conditions")

# Sidebar configuration
with st.sidebar:
    st.header("Model Settings")
    llm_model = st.selectbox("Select Open Source model", ["llama3.2"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, 
                           help="Higher values make output more creative, lower values more deterministic")
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=200,
                          help="Maximum length of the response")
    
    # Add a button to clear chat history
    if st.button("Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep only the system message
        st.session_state.enough_info = False
        st.session_state.final_assessment = None
        st.session_state.recommended_doctors = {}
        st.rerun()


def generate_response(messages, llm_model, temperature, max_tokens, is_assessment=False):
    # Instead of passing max_tokens directly to Ollama constructor, use the appropriate parameter
    # or configure it properly in the model arguments
    llm = Ollama(model=llm_model, temperature=temperature)
    
    # Create the proper model parameters depending on what Ollama expects
    model_kwargs = {"num_predict": max_tokens}  # Ollama uses num_predict instead of max_tokens
    
    # Convert messages to format LangChain can use
    langchain_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            langchain_messages.append(("system", msg.content))
        elif isinstance(msg, HumanMessage):
            langchain_messages.append(("user", msg.content))
        elif isinstance(msg, AIMessage):
            langchain_messages.append(("assistant", msg.content))
    
    # If this is a request for final assessment, add special instruction
    if is_assessment:
        assessment_instruction = """
        Based on the conversation so far, provide a structured final assessment with the following sections:
        
        1. SYMPTOM SUMMARY: Summarize all the symptoms mentioned by the user
        2. POTENTIAL CONDITIONS: List the most likely conditions in order of probability (most likely first)
        3. SEVERITY ASSESSMENT: Rate the overall urgency (Low, Medium, High)
        4. KEY RECOMMENDATIONS: Provide specific next steps the patient should take
        
        Format each section with clear headings and concise content. End with the standard medical disclaimer.
        """
        langchain_messages.append(("user", assessment_instruction))
    
    prompt = ChatPromptTemplate.from_messages(langchain_messages)
    
    # Pass the model_kwargs to the generate method
    chain = prompt | llm.bind(model_kwargs=model_kwargs) | StrOutputParser()
    response = chain.invoke({})
    return response

# Create PDF report function
def create_pdf_report(assessment_text, recommended_doctors=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Medical Symptom Assessment Report", ln=True, align='C')
    pdf.ln(10)
    
    # Add date
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Add assessment content
    pdf.set_font("Arial", size=12)
    
    # Split text into lines and add to PDF
    lines = assessment_text.split('\n')
    for line in lines:
        if line.strip() and line.strip().isupper():  # Section headers
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(5)
            pdf.cell(200, 10, txt=line.strip(), ln=True)
            pdf.set_font("Arial", size=12)
        else:
            # Handle long lines by wrapping text
            pdf.multi_cell(0, 10, txt=line)
    
    # Add doctor recommendations if available
    if recommended_doctors:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Recommended Specialists", ln=True, align='C')
        pdf.ln(10)
        
        for specialty, doctors in recommended_doctors.items():
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt=f"{specialty} Specialists", ln=True)
            pdf.ln(5)
            
            for doctor in doctors:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt=f"{doctor['name']} - Rating: {doctor['rating']}/5", ln=True)
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, txt=f"Specialty: {doctor['specialty']}", ln=True)
                pdf.cell(200, 10, txt=f"Address: {doctor['address']}", ln=True)
                pdf.cell(200, 10, txt=f"Contact: {doctor['contact']}", ln=True)
                pdf.ln(5)
    
    # Add disclaimer at the bottom
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 10, txt="IMPORTANT DISCLAIMER:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="This is not a diagnosis. The information provided is for educational purposes only. Always consult with a qualified healthcare provider for proper diagnosis and treatment.")
    
    return pdf.output(dest='S').encode('latin1')

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="medical_assessment.pdf">Download PDF Report</a>'
    return href

# Function to extract conditions from assessment
def extract_conditions_from_assessment(assessment_text):
    conditions = []
    
    # Look for the POTENTIAL CONDITIONS section in the assessment
    potential_conditions_section = re.search(r'POTENTIAL CONDITIONS:?(.*?)(?:SEVERITY ASSESSMENT|$)', assessment_text, re.DOTALL | re.IGNORECASE)
    
    if potential_conditions_section:
        section_text = potential_conditions_section.group(1).strip()
        
        # Look for numbered or bulleted items
        condition_items = re.findall(r'(?:\d+\.|\-|\*)\s*([^:]+)(?::|$)', section_text)
        
        if condition_items:
            # Extract just the condition name from each item (before any explanation)
            for item in condition_items:
                condition = item.strip().split('-')[0].split(':')[0].strip()
                conditions.append(condition)
        else:
            # If no clear items found, try to split by newlines or commas
            lines = section_text.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    for part in parts:
                        if part.strip() and len(part.strip()) > 3:  # Ensure it's not just a number or symbol
                            conditions.append(part.strip())
    
    return conditions

# Display chat history
for message in st.session_state.messages[1:]:  # Skip the system message
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
            
            # Check if this message suggests we have enough information
            if not st.session_state.enough_info and len(st.session_state.messages) > 3:
                if "based on your symptoms" in message.content.lower() or "possible conditions" in message.content.lower():
                    st.session_state.enough_info = True

# Complete Assessment button (appears when we have enough information)
if st.session_state.enough_info and not st.session_state.final_assessment:
    if st.button("Complete Assessment & Get Final Recommendations"):
        with st.spinner("Generating comprehensive assessment..."):
            final_assessment = generate_response(
                st.session_state.messages, 
                llm_model, 
                temperature, 
                max(max_tokens * 2, 500),  # Ensure enough tokens for comprehensive assessment
                is_assessment=True
            )
            st.session_state.final_assessment = final_assessment
            
            # Extract conditions and find recommended doctors
            conditions = extract_conditions_from_assessment(final_assessment)
            if conditions:
                st.session_state.recommended_doctors = find_doctors_for_conditions(conditions)

# Display final assessment if available
if st.session_state.final_assessment:
    with st.container():
        st.markdown("## Final Assessment")
        st.markdown("---")
        st.markdown(st.session_state.final_assessment)
        st.markdown("---")
        
        # Display recommended doctors
        if st.session_state.recommended_doctors:
            st.markdown("## Recommended Specialists")
            for specialty, doctors in st.session_state.recommended_doctors.items():
                with st.expander(f"{specialty} Specialists"):
                    for doctor in doctors:
                        st.markdown(f"**{doctor['name']}** - Rating: {doctor['rating']}/5")
                        st.markdown(f"Specialty: {doctor['specialty']}")
                        st.markdown(f"Address: {doctor['address']}")
                        st.markdown(f"Contact: {doctor['contact']}")
                        st.markdown("---")
        
        # Create PDF and add download button
        pdf_bytes = create_pdf_report(st.session_state.final_assessment, st.session_state.recommended_doctors)
        st.markdown(get_pdf_download_link(pdf_bytes), unsafe_allow_html=True)
        
        # Option to export as text
        if st.button("Export as Text"):
            text_report = st.session_state.final_assessment
            
            # Add doctor recommendations to text report
            if st.session_state.recommended_doctors:
                text_report += "\n\n============================================\n"
                text_report += "RECOMMENDED SPECIALISTS\n"
                text_report += "============================================\n\n"
                
                for specialty, doctors in st.session_state.recommended_doctors.items():
                    text_report += f"{specialty} Specialists:\n"
                    text_report += "-" * len(f"{specialty} Specialists:") + "\n"
                    
                    for doctor in doctors:
                        text_report += f"* {doctor['name']} - Rating: {doctor['rating']}/5\n"
                        text_report += f"  Specialty: {doctor['specialty']}\n"
                        text_report += f"  Address: {doctor['address']}\n"
                        text_report += f"  Contact: {doctor['contact']}\n\n"
            
            st.download_button(
                label="Download Text Report",
                data=text_report,
                file_name="medical_assessment.txt",
                mime="text/plain"
            )
        
        # Option to continue the conversation
        if st.button("Continue Conversation"):
            # Add the assessment to the chat history
            st.session_state.messages.append(
                HumanMessage(content="Please provide a complete assessment of my condition based on our conversation.")
            )
            st.session_state.messages.append(
                AIMessage(content=st.session_state.final_assessment)
            )
            # Reset the final assessment display
            st.session_state.final_assessment = None
            st.rerun()

# Chat input (only show if not viewing final assessment)
if not st.session_state.final_assessment:
    user_input = st.chat_input("Describe your symptoms here...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.text("Analyzing symptoms...")
            
            # Generate response
            ai_response = generate_response(st.session_state.messages, llm_model, temperature, max_tokens)
            
            # Add AI response to chat history
            st.session_state.messages.append(AIMessage(content=ai_response))
            
            # Display AI response
            thinking_placeholder.write(ai_response)
            
            # Check if we might have enough information now
            if not st.session_state.enough_info and len(st.session_state.messages) > 3:
                if "based on your symptoms" in ai_response.lower() or "possible conditions" in ai_response.lower():
                    st.session_state.enough_info = True
                    st.rerun()  # Refresh to show the Complete Assessment button

# # Display helpful information
# with st.expander("About this Medical Symptom Analyzer"):
#     st.markdown("""
#     ### How to use this tool:
#     1. Describe your symptoms in detail in the chat input
#     2. Be specific about duration, severity, and any triggers
#     3. Answer follow-up questions from the AI to help narrow down possibilities
#     4. When enough information is gathered, you'll see a "Complete Assessment" button
#     5. The final assessment will include recommended specialists for your condition
#     6. The complete report can be downloaded as PDF or text for your records
    
#     ### Important Disclaimer:
#     This tool is for informational purposes only and does not replace professional medical advice.
#     Always consult with a qualified healthcare provider for proper diagnosis and treatment.
#     """)