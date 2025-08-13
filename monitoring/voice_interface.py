import speech_recognition as sr
import pyttsx3
from typing import Dict, Callable
import asyncio
import json
class VoiceInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.commands: Dict[str, Callable] = {}
        self.listening = False
    def register_command(self, keyword: str, handler: Callable):
        self.commands[keyword.lower()] = handler
    async def start_listening(self):
        self.listening = True
        while self.listening:
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio).lower()
                    await self._process_command(text)
            except Exception as e:
                print(f"Error: {e}")
            await asyncio.sleep(0.1)
    async def _process_command(self, text: str):
        for keyword, handler in self.commands.items():
            if keyword in text:
                try:
                    response = await handler(text)
                    self.speak(response)
                except Exception as e:
                    self.speak(f"Error executing command: {str(e)}")
                return
        self.speak("Command not recognized")
    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
    def stop(self):
        self.listening = False
