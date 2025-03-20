from langchain_community.retrievers import PubMedRetriever
from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List, Dict, Any, Set
from temp.temp_browser import scrape_page_with_doi, login
import os
import shutil
import json
import hashlib
import datetime
import re
from langchain_docling.loader import DoclingLoader
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

async def search_pubmed(query: str) -> List[Document]:
    """PubMed에서 논문을 검색합니다."""
    retriever = PubMedRetriever(top_k_results=5)
    return await retriever.ainvoke(query)

def sanitize_filename(filename: str, max_length: int = 80) -> str:
    """
    파일 이름에 사용할 수 없는 문자를 제거하고 길이를 제한합니다.
    
    Args:
        filename: 원본 파일 이름
        max_length: 최대 파일 이름 길이 (확장자 제외)
        
    Returns:
        안전한 파일 이름
    """
    # 파일 이름에 사용할 수 없는 문자 제거
    safe_name = re.sub(r'[\\/*?:"<>|,]', "_", filename)
    
    # 공백을 언더스코어로 변환
    safe_name = safe_name.replace(" ", "_")
    
    # 길이 제한
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name

async def scrape_article(docs: List[Document]) -> List[str]:
    """DOI를 입력받아 해당 페이지를 스크래핑합니다."""
    
    processed_html_folder = "processed_html"
    recent_html_folder = "recent_html"

    if not os.path.exists(recent_html_folder):
        os.makedirs(recent_html_folder)
        print(f"'{recent_html_folder}' 폴더가 생성되었습니다.")

    if not os.path.exists(processed_html_folder):
        os.makedirs(processed_html_folder)
        print(f"'{processed_html_folder}' 폴더가 생성되었습니다.")

    async def scrape_single_article(doc):
        title = doc.metadata.get("Title", "Unknown Title")
        doi = doc.metadata.get("DOI", "")
        
        # 파일 이름 안전하게 변환
        safe_title = sanitize_filename(title)
        
        if not doi:
            print(f"문서 '{title}'에 DOI가 없습니다. 스크래핑을 건너뜁니다.")
            return None
        
        file_path = os.path.join(processed_html_folder, f"{safe_title}.html")
        if os.path.exists(file_path):
            print(f"'{file_path}' 파일이 이미 존재합니다. 스크래핑을 건너뜁니다.")
            return None
        
        content = await scrape_page_with_doi(doi)
        if content:
            file_path = os.path.join(recent_html_folder, f"{safe_title}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))
            
            # 메타데이터 파일에 원래 제목도 저장
            metadata_path = os.path.join(recent_html_folder, f"{safe_title}.metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                metadata = {
                    "Title": title,  # 원래 제목
                    "DOI": doi,
                    "safe_title": safe_title
                }
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"'{title}' 문서를 성공적으로 스크래핑했습니다.")
            return file_path
        else:
            print(f"'{title}' 문서 스크래핑 결과가 비어있습니다.")
            return None
    
    # 모든 문서를 병렬로 스크래핑
    import asyncio
    scrape_tasks = [scrape_single_article(doc) for doc in docs]
    results = await asyncio.gather(*scrape_tasks)
    
    # None 값 필터링
    return [result for result in results if result]

def generate_doc_id(doi: str) -> str:
    """DOI 기반으로 고유 문서 ID를 생성합니다."""
    return hashlib.md5(doi.encode()).hexdigest()

def extract_compact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """docling 메타데이터에서 핵심 정보만 추출합니다."""
    compact = {}
    important_fields = ["title", "authors", "abstract", "keywords", "publication_date", "journal"]
    
    for field in important_fields:
        if field in metadata:
            compact[field] = metadata[field]
    
    return compact

def embedding_article(file_path_list: List[str]):
    """HTML 파일을 처리하여 마크다운으로 변환하고 Chroma 벡터 스토어에 저장합니다."""
    # 결과를 저장할 디렉토리 설정
    vector_store_dir = "vector_store"
    processed_html_folder = "processed_html"
    
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)
        print(f"'{vector_store_dir}' 폴더가 생성되었습니다.")
    
    if not os.path.exists(processed_html_folder):
        os.makedirs(processed_html_folder)
        print(f"'{processed_html_folder}' 폴더가 생성되었습니다.")
    
    # 처리된 문서 ID를 저장할 세트 초기화
    processed_docs = set()
    
    # 기존 처리된 문서 ID 로드 (파일이 있는 경우)
    processed_docs_file = os.path.join(vector_store_dir, "processed_docs.json")
    if os.path.exists(processed_docs_file):
        try:
            with open(processed_docs_file, "r", encoding="utf-8") as f:
                processed_docs = set(json.load(f))
        except Exception as e:
            print(f"처리된 문서 목록 로드 실패: {e}")
    
    # 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.getenv("OPENAI_API_KEY"))
    
    # Chroma 벡터 스토어 초기화
    vector_store = Chroma(
        collection_name="pubmed_articles",
        persist_directory=vector_store_dir,
        embedding_function=embedding_model
    )
    
    all_chunks = []
    processed_files = []
    newly_processed_docs = set()
    
    # 마크다운 헤더 분할기 설정
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    for file_path in file_path_list:
        try:
            base_filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            
            # 메타데이터 파일 경로
            metadata_path = os.path.join(os.path.dirname(file_path), f"{filename_without_ext}.metadata.json")
            
            # 문서 메타데이터 로드
            doc_metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    try:
                        doc_metadata = json.load(f)
                    except:
                        print(f"메타데이터 파일 '{metadata_path}' 로드 실패")
            
            # DOI 기반 고유 ID 생성
            doc_id = None
            if "doi" in doc_metadata:
                doc_id = generate_doc_id(doc_metadata["DOI"])
                
                # 이미 벡터 스토어에 저장된 문서인지 확인
                if doc_id in processed_docs:
                    print(f"문서 '{filename_without_ext}'는 이미 벡터 스토어에 저장되어 있습니다. 건너뜁니다.")
                    # 파일은 여전히 processed_html로 이동시키기 위해 목록에 추가
                    processed_files.append(file_path)
                    if os.path.exists(metadata_path):
                        processed_files.append(metadata_path)
                    continue
            
            print(f"{file_path} 처리 중...")
            
            # DoclingLoader를 사용하여 HTML 파일 로드
            loader = DoclingLoader(
                file_path=file_path
            )
            
            docs = loader.load()
            if not docs:
                print(f"경고: {file_path}에서 문서를 로드할 수 없습니다.")
                continue
                
            document = docs[0]
            
            # 원본 파일의 메타데이터 추출 및 보강
            docling_metadata = document.metadata.get("dl_meta", {})
            
            # 최종 메타데이터 구성
            source_metadata = {
                "source": file_path,
                "filename": base_filename,
                "filetype": "html",
                "processing_date": datetime.datetime.now().isoformat()
            }
            
            # 메타데이터 병합 (문서 메타데이터 우선)
            if doc_metadata:
                source_metadata.update(doc_metadata)
            
            # 마크다운 헤더 기반으로 문서 분할
            md_chunks = []
            try:
                # 마크다운 헤더로 분할 시도
                md_chunks = splitter.split_text(document.page_content)
                print(f"마크다운 헤더 기반 분할: {len(md_chunks)}개 청크 생성")
            except Exception as e:
                print(f"마크다운 분할 오류: {e}. HybridChunker로 대체합니다.")
                md_chunks = []
            
            if md_chunks:
                # 마크다운 헤더 분할이 성공한 경우
                for i, chunk in enumerate(md_chunks):
                    # 각 청크에 원본 메타데이터 추가
                    chunk.metadata.update(source_metadata)
                    
                    # 헤더 정보 메타데이터에 추가
                    if "header1" in chunk.metadata:
                        chunk.metadata["section"] = chunk.metadata["header1"]
                    
                    # 청크 인덱스 및 ID 추가
                    chunk.metadata["chunk_index"] = i
                    if doc_id:
                        chunk.metadata["doc_id"] = doc_id
                        chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
                    
                    all_chunks.append(chunk)
            else:
                # 마크다운 헤더 분할이 실패한 경우 HybridChunker 사용
                chunker = HybridChunker(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                # 문서 청킹
                chunks = chunker.split_document(document)
                
                print(f"HybridChunker 분할: {len(chunks)}개 청크 생성")
                
                # 청크 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    # 원본 메타데이터 병합
                    chunk.metadata.update(source_metadata)
                    
                    # docling 메타데이터가 있으면 추가
                    if docling_metadata:
                        # 주요 메타데이터만 추출하여 저장 (전체 메타데이터는 너무 클 수 있음)
                        compact_metadata = extract_compact_metadata(docling_metadata)
                        chunk.metadata["docling_metadata"] = compact_metadata
                    
                    # 청크 인덱스 및 ID 추가
                    chunk.metadata["chunk_index"] = i
                    if doc_id:
                        chunk.metadata["doc_id"] = doc_id
                        chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
                    
                    all_chunks.append(chunk)
            
            # 처리된 파일 목록에 추가
            processed_files.append(file_path)
            if os.path.exists(metadata_path):
                processed_files.append(metadata_path)
                
            # 처리된 문서 ID 추가
            if doc_id:
                newly_processed_docs.add(doc_id)
            
        except Exception as e:
            print(f"{file_path} 처리 중 오류 발생: {e}")
    
    # 벡터 스토어에 청크 추가
    if all_chunks:
        print(f"벡터 스토어에 {len(all_chunks)}개 청크 추가 중...")
        vector_store.add_documents(all_chunks)
        print(f"총 {len(all_chunks)}개의 청크가 벡터 스토어에 저장되었습니다.")
        
        # 처리된 문서 ID 기록 업데이트
        processed_docs.update(newly_processed_docs)
        
        # 처리된 문서 ID 저장
        try:
            with open(processed_docs_file, "w", encoding="utf-8") as f:
                json.dump(list(processed_docs), f)
            print(f"처리된 문서 ID 목록이 {processed_docs_file}에 저장되었습니다.")
        except Exception as e:
            print(f"처리된 문서 ID 목록 저장 중 오류 발생: {e}")
    else:
        print("저장할 청크가 없습니다.")
    
    # 벡터 스토어 상태 출력
    collection_count = len(vector_store.get()["ids"]) if vector_store.get() else 0
    print(f"벡터 스토어 현재 상태: {collection_count}개 문서")
    
    # 처리된 파일을 recent_html에서 processed_html로 이동
    for file_path in processed_files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(processed_html_folder, file_name)
        
        try:
            shutil.move(file_path, destination_path)
            print(f"'{file_path}'를 '{destination_path}'로 이동했습니다.")
        except Exception as e:
            print(f"파일 이동 중 오류 발생: {e}")
    
    return True  # await와 함께 사용할 수 없으므로 Chroma 객체가 아닌 단순 값 반환

# 마크다운 텍스트 길이 제한 함수 (유틸리티)
def clip_text(text, threshold=100):
    """텍스트 길이를 제한하는 유틸리티 함수입니다."""
    return f"{text[:threshold]}..." if len(text) > threshold else text
