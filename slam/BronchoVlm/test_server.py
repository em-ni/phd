#!/usr/bin/env python3
"""
Test script to check if SmolVLM server is working properly
"""

import requests
import base64
import cv2
from PIL import Image
from io import BytesIO

def test_server():
    server_url = "http://localhost:8080"
    
    print("🧪 Testing SmolVLM Server")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: List models
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Available models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        print(f"   ⚠️  Models endpoint failed: {e}")
    
    # Test 3: Simple text completion
    print("\n3. Testing text completion...")
    try:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, are you working?"}],
            "max_tokens": 20
        }
        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data['choices'][0]['message']['content']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Text completion failed: {e}")
    
    # Test 4: Create a test image
    print("\n4. Testing image analysis...")
    try:
        # Create a simple test image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 200), color='blue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='red')
        draw.text((75, 75), "TEST", fill='white')
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_data = buffer.getvalue()
        base64_string = base64.b64encode(img_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_string}"
        
        payload = {
            "model": "gpt-4-vision-preview",
            "max_tokens": 50,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }
        
        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data['choices'][0]['message']['content']}")
            print("   ✅ Image analysis working!")
            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Image analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = test_server()
    print(f"\n{'✅ Server is working!' if success else '❌ Server has issues!'}")
