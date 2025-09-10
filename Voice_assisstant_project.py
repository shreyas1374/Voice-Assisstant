import os
import time
import urllib.parse
import threading

import speech_recognition as sr
import pyttsx3
import webbrowser
import requests
import datetime

import cv2
from ultralytics import YOLO

# Optional LLM dependencies
try:
    import openai
except Exception:
    openai = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None


NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

HF_MODEL_ID = "gpt2" 

YOLO_MODEL = "yolov8n.pt"

# -------------------- Text-to-speech setup --------------------
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)


def speak(text: str):
    """Speak and print text."""
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()


# -------------------- Speech-to-text --------------------
recognizer = sr.Recognizer()


def listen(timeout=5, phrase_time_limit=8) -> str:
    """Listen from microphone and return recognized text (lowercased)."""
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 0.8
        audio = None
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return ""
    try:
        text = recognizer.recognize_google(audio, language="en-in")
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return ""


# -------------------- LLM Integration --------------------

use_openai = False
if OPENAI_API_KEY and openai is not None:
    openai.api_key = OPENAI_API_KEY
    use_openai = True
    print("LLM: Using OpenAI API for Q/A.")
else:
    if AutoModelForCausalLM is not None and pipeline is not None:
        try:
            print("LLM: Setting up local HuggingFace model pipeline. Model:", HF_MODEL_ID)
            hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                device_map="auto",
                torch_dtype="auto",
            )
            hf_pipe = pipeline("text-generation", model=hf_model, tokenizer=hf_tokenizer, max_new_tokens=256, do_sample=True, temperature=0.7)
            print("LLM: Local HF pipeline ready.")
        except Exception as e:
            print("LLM: Failed to load local model. Exception:", e)
            hf_pipe = None
    else:
        hf_pipe = None
        print("LLM: No OpenAI key provided and transformers not available. LLM Q/A disabled.")


def ask_llm(question: str) -> str:

    if use_openai:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                max_tokens=300,
                temperature=0.6,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            print("OpenAI request failed:", e)
            return "Sorry, I couldn't reach the language model."
    else:
        if hf_pipe is None:
            return "LLM not configured on this machine. Provide an OpenAI API key or install a HuggingFace model."
        try:
            out = hf_pipe(question, max_new_tokens=200, do_sample=True, temperature=0.7)
            text = out[0]["generated_text"]
            if question in text:
                answer = text.split(question, 1)[1].strip()
            else:
                answer = text.strip()
            return answer if answer else "I couldn't generate a helpful answer."
        except Exception as e:
            print("HF generation failed:", e)
            return "Sorry, the local model failed to produce an answer."


# -------------------- YOLO Object Detection --------------------
try:
    yolo_model = YOLO(YOLO_MODEL)
    print("YOLO model loaded:", YOLO_MODEL)
except Exception as e:
    print("Failed to load YOLO model:", e)
    yolo_model = None


def detect_objects_from_camera(timeout_seconds=8):
    """
    Capture a frame from webcam and run YOLO detection.
    Returns a list of (label, confidence).
    """
    if yolo_model is None:
        return None, "YOLO model not available."

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None, "Could not open webcam."

    t0 = time.time()
    frame = None
    while time.time() - t0 < 1.0:
        ret, frame = cap.read()
        if not ret:
            break
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, "Could not capture a frame from the webcam."

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(source=img, save=False, imgsz=640, conf=0.25, verbose=False)
    res = results[0]

    detections = []
    try:
        for box in res.boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
            cls_idx = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
            label = yolo_model.names.get(cls_idx, str(cls_idx))
            detections.append((label, conf))
    except Exception:
        for i in range(len(res.boxes)):
            try:
                conf = float(res.boxes.conf[i])
                cls_idx = int(res.boxes.cls[i])
                label = yolo_model.names.get(cls_idx, str(cls_idx))
                detections.append((label, conf))
            except Exception:
                continue

    if not detections:
        return [], "No objects detected."

    agg = {}
    for label, conf in detections:
        if label not in agg or conf > agg[label]:
            agg[label] = conf
    agg_list = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return agg_list, None


#--------------------functions--------------------
def open_website(command):
    if "youtube" in command:
        webbrowser.open("https://www.youtube.com")
        speak("Opening YouTube")
    elif "google" in command:
        webbrowser.open("https://www.google.com")
        speak("Opening Google")
    elif "github" in command:
        webbrowser.open("https://github.com")
        speak("Opening GitHub")
    else:
        speak("Which website do you want me to open?")
        site = listen()
        if site:
            url = "https://" + site.replace(" ", "") + ".com"
            webbrowser.open(url)
            speak(f"Opening {site}")


def get_news():
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY" or not NEWS_API_KEY:
        speak("News API key not set. Please set NEWS_API_KEY environment variable or put the key in the code.")
        return
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url).json()
        articles = resp.get("articles", [])[:5]
        if not articles:
            speak("Couldn't find news at the moment.")
            return
        speak("Here are top headlines.")
        for i, a in enumerate(articles, 1):
            speak(f"{i}. {a.get('title')}")
    except Exception:
        speak("Failed to fetch news.")


def get_weather(city="Bangalore"):
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY" or not OPENWEATHER_API_KEY:
        speak("OpenWeather API key not set. Please set OPENWEATHER_API_KEY environment variable or put the key in the code.")
        return
    url = f"http://api.openweathermap.org/data/2.5/weather?q={urllib.parse.quote(city)}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        r = requests.get(url).json()
        if r.get("cod") == 200:
            temp = r["main"]["temp"]
            desc = r["weather"][0]["description"]
            speak(f"The weather in {city} is {desc} with {temp}°C")
        else:
            speak("Couldn't find weather for that city.")
    except Exception:
        speak("Failed to fetch weather.")


def play_spotify(command):
    song = command.replace("play", "").replace("spotify", "").replace("on spotify", "").strip()
    if not song:
        speak("Please tell me the song name.")
        return
    query = urllib.parse.quote(song)
    url = f"https://open.spotify.com/search/{query}"
    webbrowser.open(url)
    speak(f"Searching Spotify for {song}")


# -------------------- Main Assistant Loop --------------------
def main():
    speak("Hello! I am your advanced assistant. Ask me anything or say 'what is in front' to detect objects.")
    while True:
        command = listen()
        if not command:
            continue

        if any(x in command for x in ["exit"]):
            speak("Goodbye! Have a great day.")
            break

        if "time" in command:
            now = datetime.datetime.now().strftime("%H:%M")
            speak(f"The time is {now}")
            continue

        if "open" in command:
            open_website(command)
            continue

        if "news" in command:
            get_news()
            continue

        if "weather" in command:
            speak("Which city?")
            city = listen()
            if city:
                get_weather(city)
            continue

        if "spotify" in command or ("play" in command and "song" in command) or ("play" in command and "on spotify" in command):
            play_spotify(command)
            continue

        if any(phrase in command for phrase in ["what is in front","detect", "what do you see"]):
            speak("Okay, I will check the camera now.")
            detections, err = detect_objects_from_camera()
            if err:
                speak(err)
            else:
                if not detections:
                    speak("I couldn't detect any objects.")
                else:
                    to_speak = []
                    for label, conf in detections[:3]:
                        to_speak.append(f"{label} with {int(conf*100)} percent confidence")
                    speak("I see " + ", ".join(to_speak) + ".")
            continue

        if any(qword in command for qword in ["who", "what", "why", "how", "explain", "tell me", "define", "describe"]):
            speak("Let me think about that.")
            answer = ask_llm(command)
            speak(answer)
            continue

        speak("I don't have a direct function for that. I can answer general questions, fetch news, weather, open sites, play Spotify, and detect objects.")


if __name__ == "__main__":
    main()
