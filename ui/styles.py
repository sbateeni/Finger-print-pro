CUSTOM_CSS = """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-text {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .quality-score {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #27ae60;
        margin: 1rem 0;
    }
    .match-result {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .match-result.match {
        background-color: #2ecc71;
        color: white;
    }
    .match-result.no-match {
        background-color: #e74c3c;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    .image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""" 