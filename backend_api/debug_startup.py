
import asyncio
from main import app, lifespan

async def test_startup():
    print("Testing startup...")
    try:
        # Manually trigger lifespan
        async with lifespan(app):
            print("Lifespan entered successfully.")
    except Exception as e:
        print(f"Lifespan failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_startup())
