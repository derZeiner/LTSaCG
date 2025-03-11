import openai
import re
import os
import whisper
from moviepy import VideoFileClip
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Extract audio from video using moviepy
def extract_audio(video_file, audio_file):
    video_clip = VideoFileClip(video_file)
    video_clip.audio.write_audiofile(audio_file)

# Step 2: Transcribe audio to SRT with Whisper
def transcribe_audio(audio_file, model='large-v3'):
    model = whisper.load_model(model)
    result = model.transcribe(audio_file)
    srt_content = whisper.utils.get_writer("srt", "./").write(result, audio_file)
    srt_file = audio_file + ".srt"
    return srt_file

# Step 3: Parse SRT to timestamped transcript
def parse_srt(srt_file):
    with open(srt_file, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2}),\d+ --> .+?\n(.+?)(?:\n\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    transcript = ""
    for _, start, text in matches:
        text = text.replace('\n', ' ').strip()
        transcript += f"[{start}] {text}\n"

    return transcript

# Step 4: Generate chapters using GPT-4o
def generate_chapters(transcript):
    prompt = f"""Here is a timestamped transcript of a YouTube video:\n\n{transcript}\n\nDivide the transcript into logical chapters. For each chapter, provide a concise title along with the timestamp where the chapter begins in the format:\n\n00:00 - Chapter Title"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    chapters = response.choices[0].message.content
    return chapters

# Main pipeline
def main():
    video_folder = "videos"
    transcript_folder = "transcripts"
    done_folder = "done"
    audio_folder = "audio"

    os.makedirs(transcript_folder, exist_ok=True)
    os.makedirs(done_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            base_name = os.path.splitext(video_file)[0]

            audio_file = f"{base_name}.mp3"
            audio_file = os.path.join(audio_folder, audio_file)

            print(f"Processing {video_file}...")

            extract_audio(video_path, audio_file)
            srt_file = transcribe_audio(audio_file)

            transcript = parse_srt(srt_file)
            transcript_path = os.path.join(transcript_folder, f"{base_name}_transcript.txt")
            with open(transcript_path, 'w', encoding='utf-8') as file:
                file.write(transcript)

            chapters = generate_chapters(transcript)
            chapters_path = os.path.join(done_folder, f"{base_name}_chapters.txt")
            with open(chapters_path, 'w', encoding='utf-8') as file:
                file.write(chapters)

            print(f"Finished {video_file}. Chapters saved to {chapters_path}")

# Run the script
if __name__ == "__main__":
    main()
