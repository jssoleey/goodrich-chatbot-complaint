from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from functools import lru_cache
import streamlit as st
import os
from dotenv import load_dotenv

# ======================== 설정 ========================
#load_dotenv(dotenv_path=".envfile", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======================== 전역 저장소 ========================
store = {}

# ======================== 전역 프롬프트 ========================
SYSTEM_PROMPT_SCRIPT = (
    "당신은 보험 민원 대응을 전문으로 하는 AI 상담 지원 도우미입니다.\n"
    "상담원이 입력한 민원 상황과 고객 감정 상태를 바탕으로, 고객의 불만을 효과적으로 완화하고 신뢰를 줄 수 있는 "
    "**맞춤형 응대 스크립트**와 실무에 도움이 되는 **상담 TIP**을 함께 제공하세요.\n"
    "응대 스크립트와 상담TIP 사이 구분선을 추가해서 내용을 구분해주세요.\n\n"

    "[응대 스크립트 작성 지침]\n"
    "1. 고객의 불만 사항에 대해 **구체적이고 진정성 있는 사과**를 하세요.\n"
    "2. 고객 감정 상태(1~5단계)에 따라 **공감과 진정 멘트**를 상황에 맞게 여러 번 배치하세요.\n"
    "3. 해결 방안은 형식적인 표현이 아니라, 실제 상담원이 사용할 수 있도록 **구체적인 절차나 진행 방식**을 언급하세요.\n"
    "   - 예: '계약 당시 녹취 기록 검토', '담당 부서와 협의 후 예상 소요 시간 안내'\n"
    "4. 같은 표현의 반복을 피하고, **다양한 어휘**로 공감과 책임감을 전달하세요.\n"
    "5. 스크립트는 **문단 구분**을 위해 내용 흐름에 따라 **두 번 줄바꿈(\\n\\n)**을 사용해 가독성을 높이세요.\n"
    "6. 이모지 사용 금지.\n"
    "7. 구어체지만 신뢰감 있는 톤 유지.\n\n"
    
    "[상담 TIP 작성 지침]\n"
    "- 생성된 스크립트와 관련된 구체적인 대응 팁을 2~3가지 제공하세요.\n"
    "- 팁은 고객 감정 관리, 내부 절차 설명 요령, 신뢰 회복 전략 등을 포함하세요.\n\n"

    "[출력 형식]\n"
    "- 민원인 이름을 자연스럽게 대화 속에 포함하세요.\n"
    "- 문장은 실제 전화 통화에서 사용할 수 있도록 **구어체**로 작성하세요.\n"
    "- 이중 존댓말, 과도한 격식 표현은 피하고 자연스러운 상담 말투를 사용하세요.\n\n"
    "---\n\n"   # ✅ 구분선 추가!
    "**상담 TIP**\n"
    "📌 상담 TIP\n"
    "▶️ (구체적인 팁 1)\n"
    "▶️ (구체적인 팁 2)\n"
    "▶️ (필요 시 팁 3)\n\n"

    "[입력 예시]\n"
    "- 민원인 이름: 김철수\n"
    "- 민원 내용: 보험금 지급 지연, 3회 문의, 담당자 연결 요청\n"
    "- 고객 감정 상태: 4 (화남)\n\n"
)

SYSTEM_PROMPT_CHATBOT = (
    "당신은 보험 민원 대응을 지원하는 AI 상담 도우미입니다.\n"
    "상담원이 생성된 응대 스크립트를 바탕으로 추가 질문이나 요청을 하면, "
    "**현실적이고 실무에 도움이 되는 보완 멘트와 상담 전략**을 제공합니다.\n\n"

    "[역할]\n"
    "- 상담원이 민원 대응 중 겪는 다양한 상황에 맞춰 적절한 멘트와 대응법을 제안하세요.\n"
    "- 단순히 문장을 생성하는 것을 넘어서, 왜 그런 멘트가 효과적인지 **간단한 설명**도 함께 제공하세요.\n"
    "- 고객 감정 관리, 민원 심화 대응, 대화 흐름 유지 방법 등을 적극적으로 안내하세요.\n"
    "- 상담원이 요청 시, 변경된 상황을 반영하여 **새로운 응대 스크립트**를 작성할 수도 있습니다.\n\n"

    "[답변 지침]\n"
    "1. 상담원이 요청한 상황에 맞는 **구체적인 멘트**를 제시하세요.\n"
    "2. 멘트는 실제 전화 상담에서 바로 사용할 수 있도록 **자연스럽고 신뢰감 있는 구어체**로 작성하세요.\n"
    "3. 멘트 제시 후, 해당 멘트의 활용 방법이나 주의사항을 간단히 설명하세요.\n"
    "4. 고객 감정 상태가 심각할수록, **공감과 진정 멘트**를 우선 제안하세요.\n"
    "5. 법적 대응, 책임자 요구 등 민감한 상황에서는 회사 정책을 존중하는 범위 내에서 신중한 멘트를 작성하세요.\n"
    "6. 요청이 모호할 경우, 상담원이 활용할 수 있는 **추천 멘트 예시**와 함께 대응 전략을 안내하세요.\n"
    "7. 이모지는 사용하지 마세요.\n\n"
    "8. 상담원이 추가 질문을 할 때는 반드시 **현재 상담 스크립트 내용을 충분히 참고**하여 답변하세요.\n"
    "   - 기존 스크립트와 중복되지 않도록 보완 멘트를 작성하세요.\n"
    "   - 스크립트의 톤, 표현 방식을 일관되게 유지하세요.\n"
    "   - 필요 시, 기존 스크립트의 어떤 부분과 연결되는 멘트인지 고려하세요.\n"

    "[형식 지침]\n"
    "- 멘트는 다음 형식을 지키세요:\n"
    "**👉 보완 멘트 예시**\n"
    "> \"여기에 실제 상담 멘트를 작성하세요.\"\n\n"
    "- 멘트 아래에는 간단한 **활용 팁**을 작성하세요.\n"
    "- 예시:\n"
    "**👉 보완 멘트 예시**\n"
    "> \"명수님, 충분히 이해합니다. 말씀하신 부분은 제가 끝까지 책임지고 확인하겠습니다.\"\n\n"
    "_이 멘트는 고객의 감정을 진정시키고 상담원의 책임감을 강조하는 효과가 있습니다._\n\n"

    "질문을 입력받으면 위 지침에 따라 상담원이 실무에서 바로 활용할 수 있는 답변을 제공하세요."
)

# ======================== 모델 호출 ========================
@lru_cache(maxsize=1)
def get_llm(model='gpt-4.1-mini'):
    return ChatOpenAI(
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ======================== 세션 관리 ========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ======================== 스크립트 생성 ========================
def get_script_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_SCRIPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{complaint_info}")
    ])
    return prompt | get_llm() | StrOutputParser()

def get_script_response(name, situation, emotion_level, session_id="complaint_session"):
    try:
        # 1️⃣ 고객 감정 상태 설명 매핑
        emotion_labels = {
            1: "평온",
            2: "다소 불만",
            3: "불만",
            4: "화남",
            5: "매우 화남"
        }
        emotion_desc = f"{emotion_level} ({emotion_labels.get(emotion_level, '불만')})"

        # 2️⃣ 입력 정보를 LLM에게 전달할 포맷으로 구성
        complaint_info = (
            f"- 민원인 이름: {name}\n"
            f"- 민원 내용: {situation}\n"
            f"- 고객 감정 상태: {emotion_desc}"
        )

        # 3️⃣ 체인 호출
        chain = RunnableWithMessageHistory(
            get_script_chain(),
            get_session_history,
            input_messages_key="complaint_info",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"complaint_info": complaint_info},
            config={"configurable": {"session_id": session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 민원 응대 스크립트 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 대화 챗봇 ========================
def get_chatbot_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_CHATBOT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return prompt | get_llm() | StrOutputParser()


def get_chatbot_response(user_message, script_context="", session_id="chatbot_session"):
    try:
        full_input = (
            "[주의] 아래 상담 스크립트 내용을 반드시 참고하여 상담원의 요청에 답변하세요.\n\n"
            "[현재 상담 스크립트]\n"
            f"{script_context}\n\n"
            "[상담원의 질문]\n"
            f"{user_message}"
        )

        chain = RunnableWithMessageHistory(
            get_chatbot_chain(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"input": full_input},
            config={"configurable": {"session_id": session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 추가 질문 처리 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print(f"🔥 예외 발생 - 입력 내용: {user_message}")
        print(f"🔥 예외 상세: {e}")
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 카카오톡 문자 발송 ========================
def generate_conversation_summary(message_list):
    summary_points = []
    for message in message_list:
        if message['role'] == 'user':
            summary_points.append(f"- 상담원 요청: {message['content']}")
        elif message['role'] == 'ai' and "👉 상담 멘트 예시" in message['content']:
            lines = message['content'].split('\n')
            for line in lines:
                if line.startswith("> "):
                    summary_points.append(f"- 제안 멘트: {line[2:]}")
    return "\n".join(summary_points)
    
def get_kakao_response(script_context, message_list, session_id="kakao_session"):
    try:
        conversation_summary = generate_conversation_summary(message_list)

        dynamic_prompt = f"""
            [상담 요약]
            {script_context}

            [추가 대화 요약]
            {conversation_summary}

            ⚠️ 반드시 위 상담 요약과 추가 대화 요약 내용을 반영하여 고객 발송용 카카오톡 메시지를 작성하세요.
            
            - 당신은 보험 상담 후 고객에게 발송할 카카오톡 메시지를 작성하는 상담사입니다.
            - 상담 내용을 바탕으로 고객 성향에 맞게 다음 [출력 형식]과 [작성 지침]에 따라 총 **3가지 유형**의 메시지를 작성하세요.

            "[출력 형식]\n"
            "각 메시지는 아래 제목과 형식을 반드시 지켜 작성하세요.\n\n"

            "### 1️⃣ 전문성 강조형\n"
            "(보험 전문가로서 핵심 보장 내용과 필요한 이유를 논리적으로 전달하는 메시지)\n\n"

            "### 2️⃣ 감성형\n"
            "(따뜻하고 배려 있는 말투로, 고객의 마음을 편안하게 해주는 메시지)\n\n"

            "### 3️⃣ 실제 사례형\n"
            "(실제 보험금 지급 사례나 주변 사례를 언급하며 필요성을 자연스럽게 강조하는 메시지)\n\n"
            
            [작성 지침]            
            1. 각 메시지는 **15줄 내외**로 작성하세요.
            2. 문장은 반드시 **문장 단위로 줄바꿈**하여 가독성을 높이세요.
            2-1. 한 문장이 너무 길어져도 **적절하게 줄바꿈**하여 가독성을 높이세요.
            2-2. 내용이 바뀌는 문단은 반드시 **두 번 줄바꿈**하여 가독성을 높이세요.
            3. 고객 이름을 자연스럽게 포함하고, 상황에 맞는 맞춤형 표현을 사용하세요.
            4. 문장은 정중하면서도 부담 없는 톤으로 작성하세요.
            5. 상담한 보험의 구체적인 내용(예: 치매보험의 주요 보장, 간병보험의 활용 사례 등)을 간단히 언급하세요.
            6. 고객이 이해하기 쉽게, 너무 추상적인 표현은 피하고 **실질적인 도움이 되는 설명**을 포함하세요.
            7. 상담한 보험 종류, 보완이 필요한 내용, 고객이 관심을 보인 내용용 등을 반영하세요.
            8. 가입을 강요하지 말고, '편하게 문의 주세요'와 같은 표현으로 마무리하세요.
            9. 이모지는 과하지 않게 사용해주세요.

            "[입력 예시]\n"
            "- 고객 이름: 박진호\n"
            "- 상담 요약: 치매보험과 간병보험 설명, 기존 실비보험 보장 부족 보완 제안\n"
            "- 추가 안내: 카카오톡으로 상담 내용 전달, 문의 시 추가 설명 가능\n\n"

            "[출력 예시 - 일부]\n"
            "### 2️⃣ 감성형\n"
            
            박진호님, 안녕하세요.  
            오늘 상담 드리면서 진호님께서 나누어주신 이야기 덕분에 많은 생각을 하게 되었습니다.

            배우자 분을 간병하시면서 얼마나 힘든 시간을 보내셨을지, 말씀만으로도 마음이 무거워졌습니다.  
            그 과정에서 장기 보장의 중요성을 느끼셨다는 말씀에 깊이 공감했습니다.

            가족을 위해 미리 대비하려는 진호님의 마음이 정말 인상적이었고, 저도 꼭 도움이 되고 싶었습니다.

            오늘 안내드린 치매보험과 간병보험은 혹시 모를 상황에서 진호님과 가족분들에게 든든한 버팀목이 되어줄 수 있는 보장입니다.

            특히 장기 간병이나 예상치 못한 진단이 발생했을 때, 경제적인 부담을 덜어드릴 수 있도록 설계된 상품이라  
            진호님처럼 가족을 먼저 생각하시는 분들께 꼭 필요한 준비라고 생각합니다.

            무엇보다도 진호님께서 부담 없이 시작하실 수 있도록 보장 범위와 보험료를 조율해 안내드렸으니, 너무 걱정하지 않으셔도 됩니다.

            이런 준비는 서두를 필요 없이, 천천히 고민해보시고 결정하셔도 괜찮습니다.

            혹시 다시 한번 설명이 필요하시거나, 더 궁금한 점이 생기신다면 언제든 편하게 연락 주세요.

            진호님께 가장 좋은 선택이 될 수 있도록 언제든 도움드리겠습니다.

            오늘 상담 진심으로 감사드리고, 무더운 날씨에 건강 유의하세요.

            늘 진호님과 가족분들을 응원하겠습니다. 감사합니다. 😊               
        """

        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"input": "카카오톡 메시지를 생성해 주세요."},
            config={"configurable": {"session_id": session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 카카오톡 메시지 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])
