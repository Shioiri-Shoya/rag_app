import os
import pandas as pd
import streamlit as st
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever

# Google APIキーが環境変数に設定されていない場合は、デフォルトのキーを設定
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBQz_CxKyyc-9g8n5HaWBHADl6HaYIg7F4"

@st.cache_data
def load_data(csv_file_path):
    """
    CSVファイルを読み込み、Pandas DataFrameとして返す関数

    Parameters:
    csv_file_path (str): 読み込むCSVファイルのパス

    Returns:
    pd.DataFrame: CSVファイルから読み込んだデータフレーム
    """
    df = pd.read_csv(csv_file_path)
    return df

@st.cache_data
def split_text(df):
    """
    'text'列のテキストを分割して、チャンク化したドキュメントのリストを返す関数

    Parameters:
    df (pd.DataFrame): 'text'列を含むデータフレーム

    Returns:
    list: 分割されたドキュメントのリスト
    """
    text_data = df["text"].dropna().tolist()  # NaNを除外
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=514, chunk_overlap=20)
    documents = [Document(page_content=text) for text in text_data if text.strip()]
    docs = text_splitter.split_documents(documents)
    return docs

@st.cache_resource
def initialize_retrievers_and_db(_docs):
    """
    初回のみChromaデータベースとRetrieversを初期化し、セッション状態に保存

    Parameters:
    docs (list): 分割されたテキストドキュメントのリスト
    """
    embeddings = HuggingFaceEmbeddings(model_name="pkshatech/GLuCoSE-base-ja")
    
    if "db" not in st.session_state:
        if os.path.exists(".chroma_db"):
            db = Chroma(persist_directory=".chroma_db", embedding_function=embeddings)
        else:
            db = Chroma.from_documents(_docs, embeddings, persist_directory=".chroma_db")
            db.persist()
        st.session_state.db = db
    
    if "ensemble_retriever" not in st.session_state:
        embedding_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        tfidf_retriever = TFIDFRetriever.from_documents(_docs)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[embedding_retriever, tfidf_retriever],
            weights=[0.5, 0.5]
        )
        st.session_state.ensemble_retriever = ensemble_retriever

@st.cache_resource
def create_qa_chain(_memory):
    """
    初回のみ質問応答チェーンを作成し、セッション状態に保持

    Parameters:
    _memory (ConversationBufferMemory): 会話の履歴を保持するメモリ

    Returns:
    ConversationalRetrievalChain: 質問応答チェーンオブジェクト
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True, temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.ensemble_retriever,
        memory=_memory
    )
    return qa_chain

def init_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "あなたは日本語で正確に回答する役立つアシスタントです。"}]
        st.session_state.costs = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key='question',
            output_key='answer',
            return_messages=True
        )
    else:
        if st.button("チャットを消去", key="clear"):
            st.session_state.messages.clear()
            st.session_state.memory.clear()

def ask_chat_history():
    qa_chain = create_qa_chain(st.session_state.memory)
    if user_input := st.chat_input("質問を入力してください"):
        with st.spinner("Gemini が入力しています ..."):
            result = qa_chain({"question": f"{user_input} 日本語で答えてください。"}, return_only_outputs=True)
            assistant_content = result["answer"]

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

def main():
    st.title("RAG System")
    csv_file_path = "yahoo_news_articles_preprocessed.csv"
    df = load_data(csv_file_path)
    docs = split_text(df)
    initialize_retrievers_and_db(docs)
    init_messages()
    ask_chat_history()

if __name__ == "__main__":
    main()
