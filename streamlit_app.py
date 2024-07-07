import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import requests
from bs4 import BeautifulSoup
import re
import openai
import pandas as pd
import os

# Set up your OpenAI API key
openai.api_key = "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"

embedder = SentenceTransformer('all-MiniLM-L6-v2')


def generate_response(question, context, max_length=200):
    prompt = f"Question: {question}\n\nContext: {context}\n\n"
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use 'gpt-4' or 'gpt-3.5-turbo'
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_length,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()


def get_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=True)


def create_context(question, texts, embeddings, max_len=256):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)

    k = min(1, len(embeddings))  # Select only the most relevant context

    if k == 0:
        return "No relevant context found."

    top_k = torch.topk(cos_scores, k=k)  # Adjust k as needed
    context = [texts[idx] for idx in top_k.indices]

    combined_context = " ".join(context)
    input_ids = embedder.tokenizer.encode(combined_context)
    if len(input_ids) > max_len:
        truncated_context = embedder.tokenizer.decode(input_ids[:max_len])
        return truncated_context
    return combined_context


def is_relevant_context(context, question, threshold=0.3):
    question_embedding = embedder.encode([question], convert_to_tensor=True)
    context_embedding = embedder.encode([context], convert_to_tensor=True)
    cos_score = torch.nn.functional.cosine_similarity(question_embedding, context_embedding)
    return cos_score.item() > threshold


def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return clean_text(text)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
        return ""


def clean_text(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
    return sentence.strip()


def save_query_to_csv(question, response, filename="queries.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Question", "Response"])

    new_entry = pd.DataFrame({"Question": [question], "Response": [response]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(filename, index=False)


# Streamlit app
st.title("Nord Health Solutions AI Chatbot")
st.sidebar.title('🤗💬 LLM Chat App')

# URL input
url = "https://nordhealthsolutions.com/"
st.write(f"Scraping content from: {url}")

# Scrape website text and generate embeddings
text = scrape_website(url)
if text:
    st.write("Website scraped successfully!")

    # Generate embeddings for the scraped text
    embeddings = get_embeddings([text])

    query = st.text_input("Ask questions about the scraped content:")

    if st.button("Send Query"):
        context = create_context(query, [text], embeddings)
        if context == "No relevant context found." or not is_relevant_context(context, query):
            response = "For more information, feel free to reach out to us at info@nordhealthsolutions.com. We're here to help!"







        else:
            response = generate_response(query, context, max_length=256)

        st.write("Response:")
        st.markdown(response)

        # Save query and response to CSV
        save_query_to_csv(query, response)
else:
    st.error("Failed to scrape the website.")
