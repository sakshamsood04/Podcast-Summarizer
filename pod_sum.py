import os
import argparse
import json
from typing import List, Dict, Any, Optional
import yt_dlp
import tempfile
import openai
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np


load_dotenv()
class PodcastSummarizer:
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for transcription services")
        

        openai.api_key = self.api_key
        
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")

    def download_from_youtube(self, url: str, ffmpeg_location: Optional[str] = None) -> str:
        print(f"Downloading audio from YouTube URL: {url}")
        
        output_file = os.path.join(self.temp_dir, "audio.mp3")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_file.replace('.mp3', ''),
            'quiet': False,
            'no_warnings': False
        }

        if ffmpeg_location:
            ydl_opts['ffmpeg_location'] = ffmpeg_location
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            output_file = output_file.replace('.mp3', '') + '.mp3'
            print(f"Download completed: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error downloading from YouTube: {e}")
            raise

    def transcribe_audio(self, audio_file_path: str) -> str:
        print(f"Transcribing audio file: {audio_file_path}")
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text"
                )
                
            print(f"Transcription completed: {len(transcript)} characters")
            return transcript
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def preprocess_text(self, text: str) -> List[str]:

        sentences = sent_tokenize(text)
        
        clean_sentences = []
        for sentence in sentences:
            clean_sentence = ''.join(c.lower() if c.isalnum() else ' ' for c in sentence)
            clean_sentence = ' '.join(clean_sentence.split())
            if clean_sentence:
                clean_sentences.append(clean_sentence)
                
        return clean_sentences

    def sentence_similarity(self, sent1: str, sent2: str) -> float:

        stop_words = set(stopwords.words('english'))
        
        words1 = [word for word in sent1.split() if word not in stop_words]
        words2 = [word for word in sent2.split() if word not in stop_words]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]
        
        if sum(vector1) == 0 or sum(vector2) == 0:
            return 0.0
        
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:

        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(sentences[i], sentences[j])
                    
        return similarity_matrix

    def generate_summary(self, text: str, num_sentences: int = 5) -> str:

        print("Generating summary...")
    
        prompt = f"""Please summarize the following transcript in about {num_sentences} sentences. 
        Create a concise summary that captures the main points and key information, 
        but write it as a coherent paragraph in your own words rather than 
        directly extracting sentences from the original.
        
        Transcript:
        {text}
        """
        
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        
        summary = response.choices[0].message['content'].strip()
        print(f"Summary generated: {len(summary)} characters")
        return summary


    def extract_key_topics(self, text: str, num_topics: int = 5) -> List[str]:
        
        stop_words = set(stopwords.words('english'))
        
        words = [word.lower() for word in text.split() if word.isalnum()]
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        topics = [word for word, _ in sorted_words[:num_topics]]
        return topics

    def identify_speakers(self, text: str) -> Dict[str, List[str]]:

        try:
            prompt = (
                "Below is a podcast transcript. Please identify different speakers "
                "and organize the transcript by speaker. Format the output as:\n"
                "Speaker 1: [Statement]\nSpeaker 2: [Statement]\n\n"
                "Here's the transcript:\n\n" + text[:4000] 
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies speakers in podcast transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            
            speaker_text = response.choices[0].message.content
            
            speakers = {}
            current_speaker = "Unknown"
            
            for line in speaker_text.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2 and len(parts[0].strip()) < 30:  # Likely a speaker name
                        current_speaker = parts[0].strip()
                        content = parts[1].strip()
                        if current_speaker not in speakers:
                            speakers[current_speaker] = []
                        speakers[current_speaker].append(content)
            
            return speakers
            
        except Exception as e:
            print(f"Error identifying speakers: {e}")
            return self._fallback_speaker_identification(text)
    
    def _fallback_speaker_identification(self, text: str) -> Dict[str, List[str]]:
        speakers = {}
        lines = text.split('\n')
        current_speaker = "Unknown"
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and len(parts[0].strip()) < 30:  # Likely a speaker name
                    current_speaker = parts[0].strip()
                    content = parts[1].strip()
                    if current_speaker not in speakers:
                        speakers[current_speaker] = []
                    speakers[current_speaker].append(content)
            elif current_speaker != "Unknown":
                speakers[current_speaker].append(line.strip())
                
        if len(speakers) <= 1:
            speakers = {"Speaker": [p.strip() for p in text.split('\n\n') if p.strip()]}
                
        return speakers


    def summarize_from_url(self, url: str, summary_length: int = 5,include_topics: bool = True,include_speakers: bool = True,ffmpeg_location: Optional[str] = None) -> Dict[str, Any]:
        try:
            audio_file = self.download_from_youtube(url, ffmpeg_location)
            transcript = self.transcribe_audio(audio_file)
            summary = self.generate_summary(transcript, summary_length)
            
            result = {
                "url": url,
                "summary": summary,
            }
            
            if include_topics:
                topics = self.extract_key_topics(transcript)
                result["key_topics"] = topics
                
            if include_speakers:
                speakers = self.identify_speakers(transcript)
                result["speakers"] = speakers
                
            return result
            
        except Exception as e:
            print(f"Error processing URL: {e}")
            return {"error": str(e)}
        finally:
            self._cleanup()

    def _cleanup(self):
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up: {e}")

    def save_summary(self, summary_data: Dict[str, Any], output_file: str) -> None:

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.endswith('.json'):
                    json.dump(summary_data, f, indent=2)
                else:
                    f.write("# Podcast Summary\n\n")
                    
                    if "url" in summary_data:
                        f.write(f"Source: {summary_data['url']}\n\n")
                    
                    f.write("## Summary\n\n")
                    f.write(summary_data.get("summary", "No summary available"))
                    f.write("\n\n")
                    
                    if "key_topics" in summary_data:
                        f.write("## Key Topics\n\n")
                        for topic in summary_data["key_topics"]:
                            f.write(f"- {topic}\n")
                        f.write("\n")
                        
                    if "speakers" in summary_data:
                        f.write("## Speaker Contributions\n\n")
                        for speaker, statements in summary_data["speakers"].items():
                            f.write(f"### {speaker}\n\n")
                            for statement in statements[:3]:
                                f.write(f"- {statement}\n")
                            f.write("\n")
            
            print(f"Summary saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving summary: {e}")


def main():
    parser = argparse.ArgumentParser(description="AI Podcast Summarizer")
    parser.add_argument("url", help="YouTube or podcast URL")
    parser.add_argument("--api-key", help="OpenAI API key for transcription")
    parser.add_argument("--ffmpeg-location", help="Path to FFmpeg binary folder")
    args = parser.parse_args()
    
    summarizer = PodcastSummarizer(api_key=args.api_key)
    
    summary_data = summarizer.summarize_from_url(
        url=args.url,
        summary_length=args.sentences,
        include_topics=not args.no_topics,
        include_speakers=not args.no_speakers,
        ffmpeg_location=args.ffmpeg_location
    )
    summarizer.save_summary(summary_data, args.output)

if __name__ == "__main__":
    main()