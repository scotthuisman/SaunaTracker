import streamlit as st

from utils.llm_agent import AnalystError, chat


def render(data, unit, host, model, reference, today, consent):
    st.subheader("Ask about your history")
    st.caption(
        "Chat uses the full active dataset, independent of the chart date filter, unless your question specifies a period. Numerical results are computed in Python. Research context comes from the bundled reviewed snapshot."
    )
    if data is None:
        st.info("Upload a valid CSV to start your analysis.")
        return
    st.write(
        "Try: ‘Summarize my sauna usage for September’, ‘Which month had the most sessions?’, or ‘How have my pre- and post-sauna readings changed over time?’"
    )
    if not host:
        st.info(
            "Set a local OLLAMA_HOST and start Ollama to enable AI. All other analytics work without it."
        )
    st.session_state.setdefault("chat_history", [])
    if st.button("Clear chat"):
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input(
        "Ask Sauna Fitness Guru", disabled=not host or not consent, max_chars=4000
    )
    if question:
        history = list(st.session_state.chat_history)
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking your session history…"):
                try:
                    answer = chat(data, unit, question, history, host, model, reference, today)
                    st.write(answer)
                    st.session_state.chat_history.extend(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                except AnalystError as exc:
                    st.error(str(exc))
    if reference.get("verified"):
        st.markdown(
            f"Source: [FoundMyFitness — Sauna]({reference['source_url']}) · snapshot {reference['retrieval_date']}"
        )
