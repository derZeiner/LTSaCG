from moviepy import VideoFileClip
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def extract_audio(video_file, audio_file):
    video_clip = VideoFileClip(video_file)
    video_clip.audio.write_audiofile(audio_file)

def transcribe_audio(audio_file):
    audio_file = open(audio_file, 'rb')
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="de",
        response_format="srt"
    )
    return transcript


def generate_chapters(transcript):
    response = client.responses.create(
        model="gpt-4o",
        instructions="""
Create YouTube Video chapter titles optimized for engagement and keywords
read the whole transcript and dont make too many chapters depending on script length. people want precise and broad chapters. also the correct timestamps are really important
provide only those in the following format: hh:mm:ss - Chapter Title
""",
        input=f"Here is a timestamped transcript of a YouTube video:\n\n{transcript}\n\n"
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

            # save the srt
            srt_path = os.path.join(transcript_folder, f"{base_name}.srt")
            with open(srt_path, 'w', encoding='utf-8') as file:
                file.write(srt_file)

            chapters = generate_chapters(srt_file)
            chapters_path = os.path.join(done_folder, f"{base_name}_chapters.txt")
            with open(chapters_path, 'w', encoding='utf-8') as file:
                file.write(chapters)

            print(f"Finished {video_file}. Chapters saved to {chapters_path}")

# Run the script
if __name__ == "__main__":
    main()
