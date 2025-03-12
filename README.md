# 랭그래프 기반 논문 검색 및 분석 에이전트 시스템

이 프로젝트는 LangGraph를 활용하여 학술 논문을 검색하고 분석하는 에이전트 시스템을 구현합니다. 사용자의 질문을 바탕으로 관련 논문을 검색하고, 내용을 분석하여 종합적인 보고서를 생성합니다.

## 시스템 아키텍처

이 시스템은 다음과 같은 에이전트로 구성되어 있습니다:

1. **플래너 에이전트 (Planner)**: 사용자 질문을 분석하고 연구 계획 수립
2. **리서처 에이전트 (Researcher)**: PubMed에서 논문 검색 및 전문 수집
3. **분석가 에이전트 (Analyzer)**: 논문 내용 분석 및 데이터 추출
4. **작성가 에이전트 (Writer)**: 분석 결과를 바탕으로 보고서 작성
5. **검토자 에이전트 (Reviewer)**: 보고서 평가 및 피드백 제공
6. **편집자 에이전트 (Editor)**: 피드백을 반영하여 최종 보고서 편집

## 설치 방법

### 필수 요구사항
- Python 3.9 이상
- pip (Python 패키지 관리자)

### 설치 단계

1. 저장소 클론 또는 다운로드
   ```
   git clone https://github.com/yourusername/research-graph-agent.git
   cd research-graph-agent
   ```

2. 가상 환경 생성 및 활성화 (선택 사항)
   ```
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. 필요한 패키지 설치
   ```
   pip install -r requirements.txt
   ```

4. 환경 변수 설정
   `.env` 파일을 생성하고 다음과 같이 API 키를 설정합니다:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## 사용 방법

1. 메인 스크립트 실행
   ```
   python main_graph.py
   ```

2. 연구하고 싶은 주제나 질문을 입력합니다.

3. 시스템이 자동으로 다음 단계를 수행합니다:
   - 질문 분석 및 연구 계획 수립
   - 관련 논문 검색 및 수집
   - 논문 내용 분석 및 데이터 추출
   - 보고서 작성
   - 보고서 평가 및 편집

4. 최종 보고서는 `final_report.md` 파일로 저장됩니다.

## 파일 구조

- `main_graph.py`: 메인 실행 스크립트
- `research_graph.py`: 에이전트 및 워크플로우 구현
- `requirements.txt`: 필요한 패키지 목록
- `plan.md`: 시스템 설계 계획
- `.env`: 환경 변수 파일 (API 키 등)

## 주요 기능

- PubMed 논문 검색 및 메타데이터 추출
- 논문 전문 스크래핑 및 저장
- 논문 내용 임베딩 및 분석
- 데이터 추출 및 통계 분석
- 종합 보고서 생성 및 평가

## 제한 사항

- 현재 버전은 PubMed 데이터베이스만 지원합니다.
- 논문 전문 스크래핑은 일부 출판사에서 제한될 수 있습니다.
- 대용량 논문 처리 시 API 비용이 발생할 수 있습니다.

## 향후 개발 계획

- 추가 학술 데이터베이스 지원 (Scopus, Web of Science 등)
- 논문 간 관계 분석 및 시각화 기능
- 웹 인터페이스 개발
- 다국어 지원

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 