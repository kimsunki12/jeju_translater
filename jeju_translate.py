import pysqlite3
import sys
# 표준 라이브러리 sqlite3 대신 pysqlite3를 사용하도록 교체
sys.modules["sqlite3"] = pysqlite3

import os
import time
import streamlit as st

# pysqlite3 교체가 끝난 후에 langchain_chroma 임포트
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ✅ 환경 변수 로드
SOLAR_API_KEY = os.getenv("UPSTAGE_API_KEY")

# ✅ 제주어-한국어 병렬 데이터 로드
def load_parallel_data(jeju_path, korean_path, sample_size=500):  # ✅ 샘플 크기 제한 추가
    jeju_sentences = open(jeju_path, "r", encoding="utf-8").readlines()[:sample_size]
    korean_sentences = open(korean_path, "r", encoding="utf-8").readlines()[:sample_size]
    
    return [
        Document(page_content=jeju.strip(), metadata={"translation": ko.strip()})
        for jeju, ko in zip(jeju_sentences, korean_sentences)
    ]

# ✅ 벡터스토어 생성 (Chroma)
@st.cache_resource
def create_vectorstore():
    start_time = time.time()
    
    print("📌 제주어-한국어 데이터 로딩 시작...")
    docs = load_parallel_data("je.train", "ko.train", sample_size=500)  # ✅ 샘플 크기 제한 적용
    print(f"✅ 데이터 로드 완료 - 소요 시간: {time.time() - start_time:.2f}초")

    print("📌 벡터스토어 생성 시작...")
    vectorstore = Chroma.from_documents(docs, UpstageEmbeddings(model="embedding-passage"))  # ✅ 작은 모델 사용
    print(f"✅ 벡터스토어 생성 완료 - 총 소요 시간: {time.time() - start_time:.2f}초")

    return vectorstore.as_retriever(k=3)

retriever = create_vectorstore()
chat = ChatUpstage(upstage_api_key=SOLAR_API_KEY, model="solar-mini")

# ✅ 번역 함수
def translate_jeju_to_korean(query):
    results = retriever.get_relevant_documents(query)
    translations = [doc.metadata["translation"] for doc in results]
    
    # ✅ Solar LLM API를 활용해 자연스러운 번역 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "제주어 문장을 한국어로 자연스럽게 번역하세요."),
        ("human", f"제주어 문장: {query}\n번역 후보: {translations}")
    ])
    response = chat.invoke({"input": prompt.format_messages()})
    return response["answer"]

# ✅ Streamlit UI 설정
st.title("Jeju Dialect Translator 🇰🇷")
st.write("제주 방언을 한국어로 번역하는 AI 시스템입니다.")

query = st.text_input("제주어 입력: ")
if query:
    result = translate_jeju_to_korean(query)
    st.write("### 번역 결과:")
    st.success(result)


