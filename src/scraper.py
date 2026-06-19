import asyncio
import os
import pandas as pd
from twikit import Client

QUERIES = [

    "Ronaldo vs Messi",    # Common critic keywords
    "Ronaldo Haaland",     # Common critic keywords
    "Ronaldo",             # neutral/general keywords
    "CR7",                 # neutral/general keywords
    "Ronaldo flop",        # Common critic keywords
    "Cristiano old",       # Common critic keywords

]

async def scrape(client: Client, query: str, count: int = 200) -> list[dict]:
    print(f"Scraping query: '{query}'...")
    results = []
    try:
        # Search for top tweets
        tweets = await client.search_tweet(query, product="Top")
        
        while tweets and len(results) < count:
            for tweet in tweets:
                results.append({
                    "tweet_id":    tweet.id,
                    "username":    tweet.user.screen_name,
                    "text":        tweet.text,
                    "created_at":  tweet.created_at,
                    "lang":        tweet.lang,
                    "likes":       tweet.favorite_count,
                    "retweets":    tweet.retweet_count,
                    "query":       query,
                })
                if len(results) >= count:
                    break
            
            if len(results) < count:
                print(f"Fetching next page for query '{query}' (currently collected: {len(results)})...")
                await asyncio.sleep(5)  # Increased delay to avoid rate limits
                tweets = await tweets.next()
                
        print(f"Successfully scraped {len(results)} tweets for query: '{query}'")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error scraping query '{query}': {e}")
        
    return results

async def main():
    # Setup client with browser TLS impersonation to bypass Cloudflare
    client = Client(language="en-US", impersonate="chrome124")
    
    # Load the cookies saved manually by the user
    cookies_path = "cookies.json"
    if not os.path.exists(cookies_path):
        print(f"Error: {cookies_path} not found. Please create it and add your auth_token and ct0 cookies.")
        return
        
    print("Loading authentication cookies...")
    try:
        client.load_cookies(cookies_path)
        print("Cookies loaded successfully.")
    except Exception as e:
        print(f"Failed to load cookies: {e}")
        return

    # Run scraping for all queries
    all_tweets = []
    output_path = "data/raw/tweets_raw.csv"
    os.makedirs("data/raw", exist_ok=True)
    
    # Load existing tweets if the file already exists (to resume/accumulate progress)
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            all_tweets = existing_df.to_dict(orient="records")
            print(f"Loaded {len(all_tweets)} existing tweets from {output_path}")
        except Exception:
            pass

    for q in QUERIES:
        tweets = await scrape(client, q, count=600) 
        if tweets:
            all_tweets.extend(tweets)
            # Convert to DataFrame, remove duplicates, and save progressively
            df = pd.DataFrame(all_tweets)
            df.drop_duplicates(subset="tweet_id", inplace=True)
            df.to_csv(output_path, index=False)
            print(f"Saved/Updated progress. Total unique tweets in {output_path}: {len(df)}")
        
        # Increased delay between queries to avoid rate limits
        await asyncio.sleep(10)

    if not all_tweets:
        print("No tweets collected. Scraping finished with 0 results.")
        return

if __name__ == "__main__":
    asyncio.run(main())
