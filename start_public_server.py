"""
Public Server Starter with ngrok Tunnel
========================================
Starts the Biostatistics RAG API and creates a public URL via ngrok.

Usage:
    python start_public_server.py

Requirements:
    - ngrok account (free): https://ngrok.com
    - Set NGROK_AUTHTOKEN in .env file

The script will:
1. Start the API server locally
2. Create ngrok tunnel to expose it publicly
3. Display the public URL and API key
"""
import os
import sys
import time
import threading
import webbrowser

# Fix encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from dotenv import load_dotenv
load_dotenv()

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def start_api_server():
    """Start the FastAPI server in a thread"""
    import uvicorn
    import api_server
    
    # Preload RAG system
    print("\n[1/3] Loading RAG system...")
    from rag_system.query_interface import BiostatisticsRAG
    api_server.rag_system = BiostatisticsRAG(verbose=True)
    
    print("\n[2/3] Starting API server...")
    uvicorn.run(
        api_server.app,
        host="127.0.0.1",
        port=8002,
        log_level="warning"
    )


def start_ngrok_tunnel():
    """Start ngrok tunnel"""
    import ngrok
    
    # Check for auth token
    auth_token = os.getenv("NGROK_AUTHTOKEN")
    if not auth_token:
        print("\n" + "="*60)
        print("⚠️  NGROK_AUTHTOKEN not found in .env")
        print("="*60)
        print("\nTo get a free ngrok auth token:")
        print("1. Sign up at: https://ngrok.com")
        print("2. Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("3. Add to .env: NGROK_AUTHTOKEN=your_token_here")
        print("\nAlternatively, the server is running locally at:")
        print("   http://localhost:8002")
        print("="*60)
        return None
    
    print("\n[3/3] Creating ngrok tunnel...")
    
    try:
        # Set auth token
        ngrok.set_auth_token(auth_token)
        
        # Create tunnel
        listener = ngrok.forward(8002, authtoken=auth_token)
        public_url = listener.url()
        
        return public_url
    except Exception as e:
        print(f"\n⚠️  ngrok tunnel failed: {e}")
        print("The server is still running locally at: http://localhost:8002")
        return None


def get_api_key():
    """Get or display the API key"""
    api_keys_file = os.path.join(os.path.dirname(__file__), ".api_keys")
    
    if os.path.exists(api_keys_file):
        with open(api_keys_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    
    # Check environment
    return os.getenv("BIOSTAT_API_KEY", "")


def main():
    print("\n" + "="*60)
    print("🧬 BIOSTATISTICS RAG - PUBLIC SERVER")
    print("="*60)
    
    # Start API server in background thread
    server_thread = threading.Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("\nWaiting for server to initialize...")
    time.sleep(5)
    
    # Check if server is running
    import socket
    max_retries = 60
    for i in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 8002))
            sock.close()
            if result == 0:
                break
        except:
            pass
        time.sleep(1)
        if i % 10 == 9:
            print(f"   Still loading... ({i+1}s)")
    
    # Start ngrok tunnel
    public_url = start_ngrok_tunnel()
    
    # Get API key
    api_key = get_api_key()
    
    # Display info
    print("\n" + "="*60)
    print("✅ SERVER IS RUNNING")
    print("="*60)
    
    print(f"\n📍 Local URL:  http://localhost:8002")
    if public_url:
        print(f"🌐 Public URL: {public_url}")
    
    print(f"\n🔑 API Key: {api_key}")
    
    print("\n📖 API Documentation:")
    print(f"   Local:  http://localhost:8002/docs")
    if public_url:
        print(f"   Public: {public_url}/docs")
    
    print("\n📝 Example Usage (curl):")
    example_url = public_url if public_url else "http://localhost:8002"
    print(f'''
curl -X POST "{example_url}/query" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: {api_key}" \\
  -d '{{"query": "What is power analysis?", "top_k": 5}}'
''')
    
    print("\n📝 Example Usage (Python):")
    print(f'''
import requests

response = requests.post(
    "{example_url}/query",
    headers={{"X-API-Key": "{api_key}"}},
    json={{"query": "ANOVA assumptions", "top_k": 5}}
)
print(response.json()["answer"])
''')
    
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")


if __name__ == "__main__":
    main()
