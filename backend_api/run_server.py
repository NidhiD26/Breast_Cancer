
import uvicorn
import sys

if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)
