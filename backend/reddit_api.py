import praw
import datetime
# from textblob import TextBlob
import pandas as pd


class SocialMediaMonitor:
    def __init__(self, client_id, client_secret, user_agent):
        # Initialize the Reddit API connection
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Target specific communities known for reporting crimes/scams
        self.target_subreddits = ['Scams', 'CyberSecurity', 'IdentityTheft', 'fraud']

    def fetch_recent_incidents(self, limit=10):
        """
        Fetches the 'limit' most recent posts from target subreddits.
        """
        print(f"📡 Connecting to Social Stream (Reddit)...")
        posts_data = []

        # Combine subreddits into one stream (e.g., "Scams+fraud")
        subreddit_query = "+".join(self.target_subreddits)
        subreddit = self.reddit.subreddit(subreddit_query)

        # Fetch 'New' posts to simulate real-time intake
        for post in subreddit.new(limit=limit):
            
            # 1. Basic Extraction
            title = post.title
            body = post.selftext[:500] + "..." # Truncate long text
            full_text = f"{title} {body}"
            
            # 2. Simple Keyword Check (The "Heuristic" Layer)
            risk_keywords = ['urgent', 'money', 'bank', 'police', 'hack', 'steal', 'wallet']
            flagged = any(word in full_text.lower() for word in risk_keywords)

            # 3. Sentiment Analysis (The "Sentiment Engine")
            # Polarity: -1 (Negative/Angry) to +1 (Positive/Happy)
            # blob = TextBlob(full_text)
            # sentiment_score = blob.sentiment.polarity
            
            # 4. Construct the Data Object
            incident = {
                'source_id': post.id,
                'platform': 'Reddit',
                'subreddit': post.subreddit.display_name,
                'timestamp': datetime.datetime.fromtimestamp(post.created_utc),
                'content': title, # Using title as the main headline
                'full_text': full_text,
                # 'sentiment_score': round(sentiment_score, 2),
                'is_flagged': flagged,
                'url': post.url
            }
            posts_data.append(incident)

        return pd.DataFrame(posts_data)

# text api -> swa

# --- CONFIGURATION (REPLACE WITH YOUR KEYS) ---
CLIENT_ID = 'YOUR_REDDIT_CLIENT_ID'
CLIENT_SECRET = 'YOUR_REDDIT_CLIENT_SECRET'
USER_AGENT = 'RiskBot'

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Initialize Monitor
        monitor = SocialMediaMonitor(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
        # 2. Get Data
        df = monitor.fetch_recent_incidents(limit=5)
        
        # 3. Display Result (Mocking the "Ingestion" phase)
        print("\n--- 🚨 RECENT DETECTED SIGNALS ---")
        if not df.empty:
            # Show only high-risk or relevant columns
            print(df[['timestamp', 'subreddit', 'content', 'is_flagged']].to_string(index=False))
            
            # Example of how you'd pass this to your Risk Model
            print(f"\n[System] Ingested {len(df)} signals. Sending to Context Engine...")
        else:
            print("No recent posts found.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Did you replace the CLIENT_ID and SECRET with real keys?")