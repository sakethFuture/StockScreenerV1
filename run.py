"""
Market Pulse Screener — launcher
Local:  python run.py  →  http://localhost:5050
Render: uses gunicorn via Procfile automatically
"""
import subprocess
import sys
import os

def check_deps():
    try:
        import flask, flask_cors, yfinance, pandas, numpy, requests
        print("✓ All dependencies present")
    except ImportError as e:
        print(f"Installing missing dependency: {e}")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "flask", "flask-cors", "yfinance", "pandas", "numpy", "requests"])

if __name__ == "__main__":
    check_deps()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Market Pulse Screener on http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    from app import app
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
