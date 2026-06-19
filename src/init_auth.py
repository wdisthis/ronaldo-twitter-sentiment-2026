import asyncio
import getpass
from twikit import Client

async def main():
    client = Client(language="en-US", impersonate="chrome124")
    
    print("--- Twitter Login via Twikit ---")
    username = input("Enter Twitter Username: ")
    email = input("Enter Twitter Email: ")
    password = getpass.getpass("Enter Twitter Password (input will be hidden): ")
    
    print("\nAttempting login...")
    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=email,
            password=password
        )
        client.save_cookies("cookies.json")
        print("Login successful! Session cookies saved to cookies.json.")
    except Exception as e:
        print(f"Login failed: {e}")
        print("Please double check your credentials and try again.")

if __name__ == "__main__":
    asyncio.run(main())
