import praw
import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class SocialMediaMonitor:
    def __init__(self, client_id, client_secret, user_agent):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        self.target_subreddits = ['Scams', 'CyberSecurity', 'IdentityTheft', 'fraud']

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    
    def similarity_score(self, text):
        
        stored_posts = []

        if stored_posts is None:
            return 1
        
        new_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(new_embedding, stored_posts)[0]

        ### total score
        # score = cosine_scores
        # return 


    def fetch_recent_incidents(self, limit=10):
        """
        Fetches the 'limit' most recent posts from target subreddits.
        """
        print(f" Connecting to Social Stream (Reddit)...")
        posts_data = []

        subreddit_query = "+".join(self.target_subreddits)
        subreddit = self.reddit.subreddit(subreddit_query)

        for post in subreddit.new(limit=limit):
            
            title = post.title
            body = post.selftext[:500] + "..." # Truncate long text
            full_text = f"{title} {body}"
            
            risk_keywords = ['urgent', 'money', 'bank', 'police', 'hack', 'steal', 'wallet']
            flagged = any(word in full_text.lower() for word in risk_keywords)
            
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

CLIENT_ID = 'YOUR_REDDIT_CLIENT_ID'
CLIENT_SECRET = 'YOUR_REDDIT_CLIENT_SECRET'
USER_AGENT = 'RiskBot'

if __name__ == "__main__":
    try:
        monitor = SocialMediaMonitor(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
        df = monitor.fetch_recent_incidents(limit=5)
        
        print("\n--- RECENT DETECTED SIGNALS ---")
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
