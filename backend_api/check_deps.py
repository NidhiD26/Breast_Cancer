
try:
    import fastapi
    import uvicorn
    import pandas
    import sklearn
    import numpy
    print("All good")
except ImportError as e:
    print(f"Missing: {e}")
except Exception as e:
    print(f"Error: {e}")
