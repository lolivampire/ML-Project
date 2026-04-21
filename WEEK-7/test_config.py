"""
Test bahwa config terbaca dengan benar.
"""

from app.config import DB_HOST, DB_PORT, DB_NAME, DEBUG, PORT

def main():
    print(f"DB_HOST: {DB_HOST}")
    print(f"DB_PORT: {DB_PORT} (type: {type(DB_PORT).__name__})")
    print(f"DB_NAME: {DB_NAME}")
    print(f"DEBUG: {DEBUG} (type: {type(DEBUG).__name__})")
    print(f"PORT: {PORT} (type: {type(PORT).__name__})")

if __name__ == "__main__": main()