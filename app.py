import os
import time
import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Configure the Gemini API
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def upload_to_gemini(path, mime_type="text/csv"):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

def wait_for_file_active(file):
    """Waits for the uploaded file to be active."""
    while file.state.name == "PROCESSING":
        time.sleep(10)
        file = genai.get_file(file.name)
    if file.state.name != "ACTIVE":
        raise Exception(f"File {file.name} failed to process")
    return file

def analyze_sentiment(file_path):
    """Performs sentiment analysis on the uploaded CSV file."""
    # Upload the file to Gemini
    uploaded_file = upload_to_gemini(file_path)
    uploaded_file = wait_for_file_active(uploaded_file)

    # Configure the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction="You are an expert sentiment analyzer. Analyze the content of the uploaded CSV file and provide sentiment percentages for positive, negative, and neutral reviews, along with the count in each category. Do not give the response in markdown format just give plain string, give the perfect integer numbers",
    )

    # Start a chat session with the uploaded file
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [uploaded_file],
            }
        ]
    )

    response = chat_session.send_message("Analyze the sentiment of this file.")
    return response.text

def parse_analysis_result(result):
    """Parses the analysis result to extract sentiment percentages."""
    try:
        # Debug: Log the raw result for analysis
        print("Raw Result:\n", result)
        
        lines = result.strip().split("\n")
        sentiment_data = {}
        
        for line in lines:
            if "Positive" in line:
                sentiment_data["Positive"] = int(line.split(":")[1].strip().replace("%", "").split()[0])
            elif "Negative" in line:
                sentiment_data["Negative"] = int(line.split(":")[1].strip().replace("%", "").split()[0])
            elif "Neutral" in line:
                sentiment_data["Neutral"] = int(line.split(":")[1].strip().replace("%", "").split()[0])
        
        # Ensure all three sentiment types are present
        if len(sentiment_data) != 3:
            raise ValueError("Incomplete sentiment data found in the result.")
        
        return sentiment_data
    except Exception as e:
        # Debug: Show which part failed
        print(f"Error during parsing: {e}")
        raise ValueError(f"Failed to parse result: {e}")



def generate_pie_chart(data):
    """Generates a pie chart for sentiment percentages."""
    labels = list(data.keys())
    sizes = list(data.values())
    colors = ["#4CAF50", "#F44336", "#FFC107"]  # Green, Red, Yellow
    explode = (0.1, 0.1, 0)  # Explode Positive and Negative slices slightly

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
        colors=colors,
    )
    ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.
    return fig

# Streamlit UI
st.title("Sentiment Analysis Tool")
st.write("Upload a CSV file containing reviews or comments to analyze the sentiment.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    with st.spinner("Analyzing sentiment..."):
        # Save the uploaded file locally
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Perform sentiment analysis
        try:
            result = analyze_sentiment(temp_file_path)
            st.success("Sentiment analysis completed!")
            st.write("### Sentiment Analysis Result:")
            st.write(result)
            
            # Parse and display pie chart
            sentiment_data = parse_analysis_result(result)
            st.write("### Sentiment Distribution:")
            pie_chart = generate_pie_chart(sentiment_data)
            st.pyplot(pie_chart)
        
        except Exception as e:
            st.error(f"Error during analysis: {e}")
        finally:
            # Clean up temporary file
            os.remove(temp_file_path)
