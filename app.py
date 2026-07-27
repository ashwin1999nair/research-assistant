import streamlit as st
import requests
import os

API_URL=os.getenv("API_URL", "http://localhost:8000") ## looks for an environment variable called API_URL else default to "http://localhost:8000"

st.title("AI Research Assistant")
st.subheader("Enter a topic of which you want to generate a report")

topic=st.text_input("Research Topic", placeholder="e.g. quantum computing")

if st.button("Generate Report"):
    if not topic.strip():
        st.error("Please enter a research topic")
    else:
        with st.spinner("Researching... this may take 1-2 minutes"):
            try:
                response=requests.post(
                    f"{API_URL}/research",
                    json={"topic": topic},
                    timeout=300
                )

                if response.status_code==200:
                    result=response.json()
                    st.success("Report Generated!")

                    st.subheader("Sources")
                    for url in result["urls"]:
                        st.write(url)

                    st.subheader("Research Report")
                    st.markdown(result["report"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Something went wrong.')}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to backend. Make sure FastAPI is running on {API_URL}.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The search is taking too long.")