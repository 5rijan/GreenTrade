import streamlit as st
import google.generativeai as genai
import pandas as pd

def chatbot(data: pd.DataFrame = None):
    """
    Displays a simple chatbot interface using Gemini Pro, optionally sending a DataFrame as context.

    Args:
        data (pd.DataFrame, optional): DataFrame to provide as context to the model. Defaults to None.
    """
    st.subheader("Ask a Question")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with the route data?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Prepare the prompt with optional DataFrame context
            if data is not None and not data.empty:
                context = f"Here is the route data for your reference:\n{data.to_string()}\n\n"
                full_prompt = context + prompt
            else:
                full_prompt = prompt

            response = model.generate_content(full_prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.chat_message("assistant").write(response.text)
        except ImportError:
            st.error("The `google-generativeai` library is not installed. Please install it using `pip install google-generativeai`.")
        except Exception as e:
            st.error(f"An error occurred while connecting to the AI model: {e}")
            st.error(f"Details: {e}")

if __name__ == '__main__':
    # Example DataFrame for testing
    test_data = pd.DataFrame({
        'Route Desc': ['Option A', 'Option B'],
        'Distance': [1000, 1500],
        'Est. Carbon (kg CO2e)': [50, 75]
    })
    chatbot(data=test_data)