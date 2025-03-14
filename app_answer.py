import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# .envファイルをロードして環境変数を設定
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。Google CloudのAPIキーを設定してください。")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_resource
def get_gemini_model():
    """
    Geminiモデルを取得する関数
    Returns:
        GenerativeModel: Geminiモデルインスタンス
    """
    try:
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Geminiモデルの取得中にエラーが発生しました: {e}")
        st.stop()

@st.cache_data
def load_data(csv_file_path):
    """
    CSVファイルを読み込む関数
    Args:
        csv_file_path (str): CSVファイルのパス
    Returns:
        pd.DataFrame: CSVファイルのデータを含むデータフレーム
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {e}")
        st.stop()

@st.cache_resource
def build_tfidf_model(texts):
    """
    テキストデータに基づいてTF-IDFモデルを構築する関数
    Args:
        texts (list of str): テキストのリスト
    Returns:
        tuple: TF-IDF行列とベクトライザ
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

@st.cache_resource
def get_embedding_model():
    """
    SentenceTransformerモデルを取得する関数
    Returns:
        SentenceTransformer: モデルインスタンス
    """
    return SentenceTransformer('pkshatech/GLuCoSE-base-ja')

@st.cache_resource
def build_embedding_model(texts):
    """
    テキストデータに基づいて埋め込みモデルを構築する関数
    Args:
        texts (list of str): テキストのリスト
    Returns:
        np.ndarray: 埋め込みベクトルの配列
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings):
    """
    クエリに基づいてハイブリッド検索を行う関数
    Args:
        query (str): 検索クエリ
        tfidf_matrix (scipy.sparse matrix): TF-IDF行列
        tfidf_vectorizer (TfidfVectorizer): TF-IDFベクトライザ
        embeddings (np.ndarray): 埋め込みベクトルの配列
    Returns:
        list of tuple: 検索結果とスコアのリスト（インデックス, スコア）
    """
    query_tfidf_vector = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()

    model = get_embedding_model()
    query_embedding = model.encode([query])
    embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()

    combined_scores = (0.5 * tfidf_scores) + (0.5 * embedding_scores)
    results = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
    return results

def init_chat_history():
    """
    チャット履歴を初期化し、セッションステートに保存する関数
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """
    チャット履歴を表示する関数
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def respond_with_gemini(query, results, texts, top_n=3):
    """
    検索結果をコンテキストとして整形し、Geminiモデルで応答を生成する関数
    """
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    context = "\n\n".join([f"結果 {i+1}: {texts[idx]}" for i, (idx, score) in enumerate(results[:top_n])])
    prompt = f"会話履歴:\n{chat_history}\n\n以下のコンテキストを基にして質問に答えてください:\n{context}\n\n質問: {query}"
    
    model = get_gemini_model()
    response = model.generate_content(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response.text})

def main():
    st.title("RAG System")

    csv_file_path = "yahoo_news_articles_preprocessed.csv"
    df = load_data(csv_file_path)
    texts = df['text'].dropna().tolist()

    tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)
    embeddings = build_embedding_model(texts)

    init_chat_history()

    if st.button("質問をクリア"):
        st.session_state.messages = []
        st.rerun()

    user_input = st.chat_input("質問を入力してください")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Gemini が入力しています ..."):
            results = hybrid_search(user_input, tfidf_matrix, tfidf_vectorizer, embeddings)
            respond_with_gemini(user_input, results, texts)
        display_chat_history()

if __name__ == "__main__":
    main()
