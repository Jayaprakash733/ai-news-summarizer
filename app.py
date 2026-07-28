import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    background: linear-gradient(to right,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.result-box{
    background:#1E1E1E;
    padding:20px;
    border-radius:15px;
    border:1px solid #333;
}

.footer{
    text-align:center;
    color:gray;
    padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='title'>📰 AI News Summarizer</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Search the latest news using Tavily and summarize it with Mistral AI</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

max_results = st.sidebar.slider(
    "Maximum Search Results",
    1,
    10,
    5
)

model_name = st.sidebar.selectbox(
    "Mistral Model",
    [
        "mistral-small-2506",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
"""
### Features

✅ Latest News Search

✅ AI Summarization

✅ Bullet Point Output

✅ Expandable Raw Search Results
"""
)

# -----------------------------
# Input
# -----------------------------
query = st.text_input(
    "🔍 Search Topic",
    value="Latest AI news of 2026"
)

search_button = st.button("🚀 Generate Summary", use_container_width=True)

# -----------------------------
# Main Logic
# -----------------------------
if search_button:

    with st.spinner("Searching latest news..."):

        search_tool = TavilySearchResults(max_results=max_results)

        llm = ChatMistralAI(
            model=model_name
        )

        prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Summarize the following news into clear bullet points.

News:
{news}
"""
        )

        chain = prompt | llm | StrOutputParser()

        news_result = search_tool.run(query)

        summary = chain.invoke(
            {
                "news": news_result
            }
        )

    st.success("Summary Generated!")

    st.markdown("## 🤖 AI Summary")

    st.markdown(
        f"""
<div class="result-box">

{summary}

</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("📰 View Raw Search Results"):
        st.write(news_result)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
"""
<div class='footer'>
Made with ❤️ using Streamlit • LangChain • Tavily • Mistral AI
</div>
""",
unsafe_allow_html=True)