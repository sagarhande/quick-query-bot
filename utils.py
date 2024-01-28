from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests

# Load variables from .env file
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL"))


def get_query_prompt(query, context):
    prompt = f"""Answer the question based on the context below. Answer should be well formatted. If the question cannot
     be answered using the information provided answer with "I don't know".

    Context: {context}

    Question: {query}

    Answer: """
    return prompt


def get_summary_prompt(context):
    prompt = f"""
    You are a webpage summarization bot that perform given task on the only below context. You can't not make up own answers.
    Answer should be well formatted as mentioned in last.
     
     Context: {context}

    Tasks:
    1. Generate a concise summary by extracting key points from the provided content.
    2. Determine the overall tone of the content from the options: Formal, Humorous, Optimistic, Assertive, Conversational, Friendly, Serious, Authoritative, Curious, Encouraging, Informative, Pessimistic.
    3. Identify the theme(s) of the content. Themes may include technology, art, news, or any relevant categories.
    
    content_summary: 
    overall_content_tone: 
    content_themes:
    """
    return prompt


def process_response_for_summarization_task(content):
    import re
    content_summary_match = re.search(r'content_summary: (.+?)\n\n', content, re.DOTALL)
    content_summary = content_summary_match.group(1) if content_summary_match else ""
    # Extract overall_content_tone
    overall_content_tone_match = re.search(r'overall_content_tone: (.+?)\n\n', content)
    overall_content_tone = overall_content_tone_match.group(1) if overall_content_tone_match else ""

    # Extract content_themes
    content_themes_match = re.search(r'content_themes: (.+)', content)
    content_themes = content_themes_match.group(1) if content_themes_match else ""

    # Create a dictionary
    return {
        "content_summary": content_summary,
        "overall_content_tone": overall_content_tone,
        "content_themes": [theme.strip() for theme in content_themes.split(',')]
    }


def get_user_dir(user_id):
    return f"./{user_id}"


def process_urls(urls: list, user_id: str):
    if not urls:
        return {}

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=2000
        )
        docs_list = []
        for url in urls:
            response = requests.get(url)
            html_content = response.content
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract text content from the HTML
            text_content = soup.get_text(separator="\n", strip=True)
            docs_list.extend(text_splitter.create_documents([text_content]))

        # embeddings = HuggingFaceEmbeddings()

        db = Chroma.from_documents(docs_list, embeddings, persist_directory=get_user_dir(user_id))
        return db

    except Exception as e:
        raise e


def summarize_webpage(user_id, url):
    if not url:
        return {}
    try:
        # text_splitter = RecursiveCharacterTextSplitter(
        #     separators=['\n\n', '\n', '.', ','],
        #     chunk_size=6000
        # )
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.content
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract text content from the HTML
        text_content = soup.get_text(separator="\n", strip=True).replace("/n", " ")
        # docs_list = text_splitter.create_documents([text_content]) # TODO: What if content exceeds prompt token limit

        prompt = get_summary_prompt(context=text_content)
        messages = [{"role": "user", "content": prompt}]
        response = openai_client.chat.completions.create(
            model=os.environ.get("GPT_MODEL"),
            messages=messages,
            temperature=0.1,
        )
        response_content = response.choices[0].message.content
        result = process_response_for_summarization_task(response_content)
        result["usage"] = vars(response.usage)
        return result
    except Exception as e:
        raise e


def chat_with_llm(query: str, user_id: str):
    try:
        # load from disk
        db = Chroma(persist_directory=get_user_dir(user_id), embedding_function=embeddings)
        docs = db.similarity_search(query=query, k=4)
        if docs:
            context = " ".join([d.page_content for d in docs]).replace("/n", " ")
            prompt = get_query_prompt(query=query, context=context)

            messages = [{"role": "user", "content": prompt}]
            response = openai_client.chat.completions.create(
                model=os.environ.get("GPT_MODEL"),
                messages=messages,
                temperature=0.1,
            )
            response_content = response.choices[0].message.content
            usage = vars(response.usage)
            return {"query": query, "answer": response_content, "usage": usage}

        else:
            return {
                "answer": f"No Knowledge-base found for user: {user_id}. Please create one.",
                "source": "",
            }
    except Exception as e:
        raise e
