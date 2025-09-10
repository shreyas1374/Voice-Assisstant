---Project Title:
Voice Assistant with LLM, APIs, and Object Detection (YOLOv8)

---Overview:
This is a Voice assisstant with many functionalities, such as opening apps, websites, telling latest news and weather, detecting objects in the camera and other important functionalities.
This project is a Voice Assistant that can perform multiple real-world tasks using speech recognition, text-to-speech, LLMs, APIs, and computer vision.
The assistant listens to your voice commands, understands them, and responds back with speech or performs actions such as opening apps, playing music, showing the weather, giving the latest news, and even detecting objects using your camera.

---Features
Voice Interaction – Uses speechRecognition (speech → text) and pyttsx3 (text → speech).
Web & App Control – Open websites (YouTube, Google, GitHub, etc.) or applications with voice commands.
Live News – Fetches the latest headlines using NewsAPI.
Weather Updates – Gives real-time weather data using OpenWeather API.
Spotify Integration – Plays songs directly on Spotify via Spotify API.
LLM Question Answering – Uses OpenAI API to answer questions in natural language.
Object Detection – Runs YOLOv8 model on live camera feed to detect objects in front of you.
Extensible Design – Modular structure for adding new skills (like Wikipedia, emails, calendars, IoT control).

---Tech Stack
Python (core language)
SpeechRecognition, pyttsx3 – speech I/
OpenAI API – question answering with LLM
NewsAPI, WeatherAPI, SpotifyAPI – external data sources
YOLOv8 (Ultralytics) – object detection with computer vision
OpenCV – camera integration

---Future Enhancements
Wikipedia Q/A support
Email and calendar integration
IoT smart home control
Context memory (remembering previous conversations)
GUI with Tkinter/Streamlit
