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
load_dotenv(dotenv_path=".envfile", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======================== 전역 저장소 ========================
store = {}

# ======================== 전역 프롬프트 ========================
SYSTEM_PROMPT_SCRIPT = (
    """
    [응대 스크립트 작성 지침]
    1. 고객의 불만 사항에 대해 **구체적이고 진정성 있는 사과**를 하세요.
    2. 고객 감정 상태(1~5단계)에 따라 **공감과 진정 멘트**를 상황에 맞게 여러 번 배치하세요.
    3. 해결 방안은 형식적인 표현이 아니라, 실제 상담원이 사용할 수 있도록 **구체적인 절차나 진행 방식**을 언급하세요.
       - 예: '계약 당시 녹취 기록 검토', '담당 부서와 협의 후 예상 소요 시간 안내'
    4. 같은 표현의 반복을 피하고, **다양한 어휘**로 공감과 책임감을 전달하세요.
    5. 스크립트는 **문단 구분**을 위해 내용 흐름에 따라 **두 번 줄바꿈(\\n\\n)**을 사용해 가독성을 높이세요.
    6. 이모지 사용 금지.
    7. 구어체지만 신뢰감 있는 톤 유지.
    
    [상담 TIP 작성 지침]
    - 생성된 스크립트와 관련된 구체적인 대응 팁을 2~3가지 제공하세요.
    - 팁은 고객 감정 관리, 내부 절차 설명 요령, 신뢰 회복 전략 등을 포함하세요.

    [출력 형식]
    - 민원인 이름을 자연스럽게 대화 속에 포함하세요.
    - 문장은 실제 전화 통화에서 사용할 수 있도록 **구어체**로 작성하세요.
    - 이중 존댓말, 과도한 격식 표현은 피하고 자연스러운 상담 말투를 사용하세요.
    ---   # ✅ 구분선 추가!

    📌 상담 TIP
    ▶️ (구체적인 팁 1)
    ▶️ (구체적인 팁 2)
    ▶️ (필요 시 팁 3)
    """
)

SYSTEM_PROMPT_CHATBOT = (
    """
    당신은 보험 민원 대응을 지원하는 AI 상담 도우미입니다.
    상담원이 생성된 응대 스크립트를 바탕으로 추가 질문이나 요청을 하면,
    **현실적이고 실무에 도움이 되는 보완 멘트와 상담 전략**을 제공합니다.

    [역할]
    - 상담원이 민원 대응 중 겪는 다양한 상황에 맞춰 적절한 멘트와 대응법을 제안하세요.
    - 단순히 문장을 생성하는 것을 넘어서, 왜 그런 멘트가 효과적인지 **간단한 설명**도 함께 제공하세요.
    - 고객 감정 관리, 민원 심화 대응, 대화 흐름 유지 방법 등을 적극적으로 안내하세요.
    - 상담원이 요청 시, 변경된 상황을 반영하여 **새로운 응대 스크립트**를 작성할 수도 있습니다.

    [답변 지침]
    1. 상담원이 요청한 상황에 맞는 **구체적인 멘트**를 제시하세요.
    2. 멘트는 실제 전화 상담에서 바로 사용할 수 있도록 **자연스럽고 신뢰감 있는 구어체**로 작성하세요.
    3. 멘트 제시 후, 해당 멘트의 활용 방법이나 주의사항을 간단히 설명하세요.
    4. 고객 감정 상태가 심각할수록, **공감과 진정 멘트**를 우선 제안하세요.
    5. 법적 대응, 책임자 요구 등 민감한 상황에서는 회사 정책을 존중하는 범위 내에서 신중한 멘트를 작성하세요.
    6. 요청이 모호할 경우, 상담원이 활용할 수 있는 **추천 멘트 예시**와 함께 대응 전략을 안내하세요.
    7. 이모지는 사용하지 마세요.
    8. 상담원이 추가 질문을 할 때는 반드시 **현재 상담 스크립트 내용을 충분히 참고**하여 답변하세요.
       - 기존 스크립트와 중복되지 않도록 보완 멘트를 작성하세요.
       - 스크립트의 톤, 표현 방식을 일관되게 유지하세요.
       - 필요 시, 기존 스크립트의 어떤 부분과 연결되는 멘트인지 고려하세요.

    [형식 지침]
    - 멘트는 다음 형식을 지키세요:
    **👉 보완 멘트 예시**
    > \"여기에 실제 상담 멘트를 작성하세요.\"
    - 멘트 아래에는 간단한 **활용 팁**을 작성하세요.
    - 예시:\n"
    **👉 보완 멘트 예시**\n"
    > \"명수님, 충분히 이해합니다. 말씀하신 부분은 제가 끝까지 책임지고 확인하겠습니다.\"
    _이 멘트는 고객의 감정을 진정시키고 상담원의 책임감을 강조하는 효과가 있습니다._

    질문을 입력받으면 위 지침에 따라 상담원이 실무에서 바로 활용할 수 있는 답변을 제공하세요.
    """
)

# ======================== 모델 호출 ========================
@lru_cache(maxsize=1)
def get_llm(model='gpt-4.1-mini'):
    return ChatOpenAI(model=model)

# ======================== 세션 관리 ========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ======================== 스크립트 생성 ========================
def get_script_response(name, situation, emotion_level):
    try:
        # 고객 감정 상태 설명 매핑
        emotion_labels = {
            1: "평온",
            2: "다소 불만",
            3: "불만",
            4: "화남",
            5: "매우 화남"
        }
        emotion_desc = f"{emotion_level} ({emotion_labels.get(emotion_level, '불만')})"

        # 입력 정보를 LLM에게 전달할 포맷으로 구성
        complaint_info = (
            f"- 민원인 이름: {name}\n"
            f"- 민원 내용: {situation}\n"
            f"- 고객 감정 상태: {emotion_desc}"
        )

        # ⭐ 상담원 이름 불러오기
        consultant_name = st.session_state.get('user_name', '상담원')

        # ⭐ dynamic_prompt 생성
        dynamic_prompt = f"""
        당신은 보험 민원 대응을 전문으로 하는 AI 상담 지원 도우미입니다.
        상담원이 입력한 민원 상황과 고객 감정 상태를 바탕으로, 고객의 불만을 효과적으로 완화하고 신뢰를 줄 수 있는 **맞춤형 응대 스크립트**와 실무에 도움이 되는 **상담 TIP**을 함께 제공하세요.
        응대 스크립트와 상담TIP 사이 구분선을 추가해서 내용을 구분해주세요.
        고객 이름과 상담원 이름을 혼동하지 말고, 반드시 각 정보에 맞게 사용하세요.

        ⚠️ 절대 지침
        - 상담원 이름은 반드시 아래 [상담원 정보]의 이름만 사용하세요.
        - 상담원 이름을 임의로 생성하거나 변경하지 마세요.
        - 고객 이름은 반드시 [고객 정보]의 이름만 사용하세요.
        - 다른 이름, 가상의 이름을 절대 생성하지 마세요.

        [상담원 정보]
        - 상담원 이름: {consultant_name}

        [고객 정보]
        {complaint_info}

        - 스크립트의 시작 부분에서는 상담원이 본인의 이름을 말하며 정중히 인사하도록 작성하세요.
        - 예시: "안녕하세요, 저는 굿리치 상담사 **{consultant_name}**입니다."
        
        {SYSTEM_PROMPT_SCRIPT}
        """

        # 3️⃣ 체인 호출
        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{complaint_info}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="complaint_info",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"complaint_info": complaint_info},
            config={"configurable": {"session_id": st.session_state.session_id}}
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


def get_chatbot_response(user_message, script_context=""):
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
            config={"configurable": {"session_id": st.session_state.session_id}}

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
    
def get_kakao_response(script_context, message_list):
    try:
        conversation_summary = generate_conversation_summary(message_list)

        dynamic_prompt = f"""
        [민원 상담 요약]
        {script_context}

        [추가 대화 요약]
        {conversation_summary}

        ⚠️ 반드시 위 **민원 상담 요약**과 **추가 대화 요약**을 반영하여, 고객에게 발송할 카카오톡 메시지를 작성하세요.

        - 당신은 민원 처리 상담 후, 고객에게 상황을 정리하고 안내 메시지를 보내는 보험사 상담사입니다.
        - 상담 내용을 기반으로 고객의 불만을 완화하고 신뢰를 회복할 수 있도록 다음 [출력 형식]과 [작성 지침]에 따라 총 **3가지 유형**의 메시지를 작성하세요.

        [출력 형식]
        각 메시지는 아래 제목과 설명을 참고하여 작성하세요.

        ### 1️⃣ 공감형
        (고객의 불편과 감정에 깊이 공감하며, 진심 어린 위로와 배려의 메시지를 전달하세요.)

        ### 2️⃣ 정중+사과형
        (격식 있고 공식적인 어조로 진심 어린 사과와 함께 처리 진행 상황을 명확하고 책임감 있게 안내하세요.)

        ### 3️⃣ 민원전문가형
        (전문적인 용어와 절차 설명을 통해 신뢰를 주고, 체계적으로 대응하고 있음을 강조하여 고객을 안심시키는 메시지)

        [작성 지침]            
        1. 각 메시지는 **15문장 내외**로 작성하세요.
        2. 문장은 **문장 단위로 줄바꿈**하여 가독성을 높이세요.
        3. 내용이 전환될 때는 **두 번 줄바꿈**으로 문단을 구분하세요.
        4. 고객 이름을 자연스럽게 포함하세요.
        5. 원 처리 상황, 현재 진행 단계, 예상 소요 시간, 추가 문의 가능 여부 등을 반드시 안내하세요.
        6. 강압적 표현은 절대 사용하지 말고, 항상 **'편하게 문의 주세요'**, **'언제든 연락 주세요'** 등의 표현으로 마무리하세요.
        7. 이모지는 [공감형]에서만 적절히 활용하고, 다른 유형에서는 사용하지 마세요.
        8. [민원전문가형]에서는 전문성을 강조하되, 고객이 이해할 수 있도록 어려운 용어는 쉽게 풀어서 설명하세요.
        9. 불안감을 유발하는 표현은 피하고, 신뢰와 안정감을 주는 표현을 사용하세요.
        10. 모든 유형에서 고객을 존중하는 어투와 배려 깊은 표현을 유지하세요.
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
        
        kakao_session_id = f"{st.session_state.session_id}_kakao"
        
        result = chain.invoke(
            {"input": "카카오톡 메시지를 생성해 주세요."},
            config={"configurable": {"session_id": kakao_session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 카카오톡 메시지 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])
