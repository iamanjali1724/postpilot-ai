
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

st.set_page_config(
    page_title="PostPilot AI",
    page_icon="✍️",
    layout="centered"
)

# Professional blue design
st.markdown("""
<style>
.stApp {
    background-color: #f3f8ff;
}
h1, h2, h3 {
    color: #1557a6;
}
.stButton > button {
    background-color: #0a66c2;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #004182;
    color: white;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

st.title("✍️ PostPilot AI")
st.subheader("Your LinkedIn Post Creation Assistant")
st.write(
    "Turn your ideas into professional LinkedIn posts "
    "with the help of AI."
)

st.divider()

topic = st.text_area(
    "What would you like to post about?",
    placeholder="Example: What I learned while studying machine learning...",
    height=140
)

col1, col2 = st.columns(2)

with col1:
    tone = st.selectbox(
        "Writing tone",
        ["Professional", "Inspirational", "Educational",
         "Conversational", "Storytelling"]
    )

    length = st.selectbox(
        "Post length",
        ["Short", "Medium", "Long"]
    )

with col2:
    audience = st.selectbox(
        "Target audience",
        ["Students", "Recruiters", "AI/ML professionals",
         "Entrepreneurs", "General audience"]
    )

    include_hashtags = st.checkbox(
        "Include relevant hashtags",
        value=True
    )

if st.button("✨ Generate LinkedIn Post", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        prompt = f"""<|im_start|>system
You are a helpful LinkedIn content writing assistant.
Write natural, useful, professional social media posts.
Do not invent personal experiences, achievements, or facts.
<|im_end|>
<|im_start|>user
Write a {length.lower()} LinkedIn post about:
{topic}

Tone: {tone}
Target audience: {audience}
Include a strong opening hook and a meaningful ending.
{"Include relevant hashtags." if include_hashtags else "Do not include hashtags."}
Return only the finished post.
<|im_end|>
<|im_start|>assistant
"""

        try:
            with st.spinner("Loading AI and writing your post..."):
                tokenizer, model = load_model()

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens={
                            "Short": 120,
                            "Medium": 220,
                            "Long": 350
                        }[length],
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )

                new_tokens = output[0][inputs["input_ids"].shape[1]:]
                post = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True
                ).strip()

            st.success("Your post is ready!")
            st.subheader("Your Generated Post")

            edited_post = st.text_area(
                "Review and edit your post",
                value=post,
                height=300,
                key="edited_post"
            )

            st.download_button(
                "⬇️ Download Post",
                data=edited_post,
                file_name="linkedin_post.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:
            st.error(
                "The AI could not generate a post. "
                "Please check your internet connection and try again."
            )
            st.caption(f"Technical details: {error}")

st.divider()
st.caption("PostPilot AI | AI-powered LinkedIn writing assistant")

