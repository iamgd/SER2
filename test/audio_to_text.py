# audio_to_text.py

import os
import speech_recognition as sr

def list_audio_files(folder="audio"):
    # List all audio files in the specified folder
    audio_files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    return audio_files

def audio_to_text():
    # List audio files in the "audio" folder
    audio_files = list_audio_files()

    if not audio_files:
        print("No audio files found in the 'audio' folder.")
        return {}

    # Display available audio files
    print("Available audio files:")
    for i, file in enumerate(audio_files, start=1):
        print(f"{i}. {file}")

    # Prompt user to choose an audio file
    choice = int(input("Enter the number of the audio file you want to convert: "))

    # Validate user choice
    if choice < 1 or choice > len(audio_files):
        print("Invalid choice.")
        return {}

    selected_file = audio_files[choice - 1]
    audio_path = os.path.join("audio", selected_file)

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Read selected audio file
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        # Recognize speech using Google Speech Recognition in English (US)
        text_en_us = recognizer.recognize_google(audio_data, language='en-US')
        
        print("English (US):", text_en_us)
        
        result = {
            'en_us': text_en_us
        }

        # Store the result in a text file
        with open("cap_result.txt", "w") as file:
            file.write(f"{text_en_us}\n")

        return result

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return {}
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return {}

# Test the function
if __name__ == "__main__":
    result = audio_to_text()
    print(result)
