import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./riddle-factory-model"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set pad_token_id
tokenizer.pad_token = tokenizer.eos_token

# Function to generate riddles
def generate_riddles(prompt="Riddle:", max_length=100, num_return_sequences=3):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        do_sample=True,  
        top_k=40,        
        top_p=0.9,      
        temperature=0.7, 
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    riddles_with_answers = [tokenizer.decode(riddle, skip_special_tokens=True) for riddle in output]
    
    result = []
    for text in riddles_with_answers:
        if "Answer:" in text:
            riddle_part, answer_part = text.split("Answer:", 1)
            first_answer = answer_part.split(".")[0].strip()
            result.append((riddle_part.strip(), first_answer))
        else:
            result.append((text.strip(), "Not generated"))
    return result

# Streamlit UI with modern styling
st.markdown(
    """
    <style>
        @keyframes glowing {
            0% {box-shadow: 0px 0px 10px rgba(255, 140, 0, 0.8);}
            50% {box-shadow: 0px 0px 30px rgba(255, 165, 0, 1);}
            100% {box-shadow: 0px 0px 10px rgba(255, 140, 0, 0.8);}
        }

        body {
            background-color: black;
            color: white;
            font-family: 'Arial', sans-serif;
        }

        .main {
            background: rgba(22, 22, 22, 0.9);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 20px rgba(255, 223, 0, 0.5);
        }

        .stButton > button {
            background: orange;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            width: 100%;
            padding: 14px;
            border: none;
            transition: 0.4s;
            animation: glowing 1.5s infinite alternate;
        }

        .stButton > button:hover {
            background: darkorange;
            transform: scale(1.05);
        }

        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: gold;
            text-shadow: 3px 3px 15px gold;
        }

        .riddle-box {
            background: linear-gradient(145deg, #222, #333);
            padding: 20px;
            border-radius: 12px;
            font-size: 22px;
            color: gold;
            font-weight: bold;
            text-align: center;
            box-shadow: 0px 0px 20px rgba(255, 215, 0, 0.8);
            transition: 0.3s;
        }

        .riddle-box:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 30px rgba(255, 215, 0, 1);
        }

        .footer {
            text-align: center;
            font-size: 16px;
            color: gold;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<h1 class="title">🤖 AI Math Riddle Generator 🎭</h1>', unsafe_allow_html=True)
st.write("Click the button below to generate **fun and challenging riddles** with answers!")

# Generate riddles button
if st.button("✨ Generate Riddles ✨"):
    riddles = generate_riddles()
    
    for idx, (riddle, answer) in enumerate(riddles, 1):
        st.markdown(f'<div class="riddle-box"><b>🧩 Riddle {idx}:</b><br>{riddle}<br><br><b>✅ Answer:</b> {answer}</div>', unsafe_allow_html=True)

st.markdown('<p class="footer">Built with ❤️ using GPT-2</p>', unsafe_allow_html=True)
