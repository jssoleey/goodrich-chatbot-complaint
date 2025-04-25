# 1. Python 기반 이미지 선택
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요 파일 복사
COPY . /app

# 4. 라이브러리 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. 환경변수 설정 (Streamlit 설정 최소화)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_PORT=8000

# 6. Streamlit 실행 명령
CMD ["streamlit", "run", "chatbot_sale.py", "--server.port=8000", "--server.address=0.0.0.0"]
