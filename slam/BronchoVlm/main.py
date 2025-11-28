#!/usr/bin/env python3
import cv2
import requests
import base64
import time
import threading
from io import BytesIO
from PIL import Image

# Configuration
LLAMA_SERVER_URL = "http://localhost:8080"
VIDEO_PATH = "output.mp4"
FRAME_SKIP = 45
MAX_TOKENS = 120
INSTRUCTION = "Analyze this bronchoscopy image. Describe: 1) Number of visible airways/branches, 2) Anatomical features visible. Be precise and concise."

class SmolVLMAnalyzer:
    def __init__(self, server_url=LLAMA_SERVER_URL):
        self.server_url = server_url
        self.session = requests.Session()
        self.last_response = "Initializing..."
        self.processing = False
        self.analysis_count = 0
        
    def check_server(self):
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def frame_to_base64(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = pil_image.resize((512, 384), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_data = buffer.getvalue()
        base64_string = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"
    
    def analyze_frame(self, instruction, image_base64):
        if self.processing:
            return None
        self.processing = True
        try:
            payload = {
                "model": "gpt-4-vision-preview",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }]
            }
            response = self.session.post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip()
                self.last_response = result
                self.analysis_count += 1
                return result
            else:
                error_msg = f"Server error {response.status_code}"
                self.last_response = error_msg
                return error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)[:50]}"
            self.last_response = error_msg
            return error_msg
        finally:
            self.processing = False
    
    def analyze_frame_async(self, instruction, image_base64):
        def worker():
            result = self.analyze_frame(instruction, image_base64)
            if result:
                print(f"\nAnalysis #{self.analysis_count}: {result}")
        if not self.processing:
            threading.Thread(target=worker, daemon=True).start()

def draw_overlay(frame, analyzer, frame_count):
    # Status
    status_color = (0, 255, 0) if analyzer.processing else (255, 255, 255)
    status = "ANALYZING" if analyzer.processing else "READY"
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_count} | Analysis: {analyzer.analysis_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # AI Response
    response = analyzer.last_response
    words = response.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) < 80:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    for i, line in enumerate(lines[:10]):
        cv2.putText(frame, line, (10, 90 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

def main():
    global INSTRUCTION
    
    analyzer = SmolVLMAnalyzer()
    
    if not analyzer.check_server():
        print("Error: llama.cpp server not running!")
        print("Start server: llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080")
        return
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    frame_count = 0
    cv2.namedWindow("Bronchoscopy Analysis", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % FRAME_SKIP == 0:
            image_base64 = analyzer.frame_to_base64(frame)
            analyzer.analyze_frame_async(INSTRUCTION, image_base64)
        
        display_frame = draw_overlay(frame, analyzer, frame_count)
        cv2.imshow("Bronchoscopy Analysis", display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
            new_instruction = input("Enter new instruction: ").strip()
            if new_instruction:
                INSTRUCTION = new_instruction
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
