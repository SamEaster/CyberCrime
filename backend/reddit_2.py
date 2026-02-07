import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import datetime
from atproto import Client
from dotenv import load_dotenv
import os

load_dotenv()
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")
model = SentenceTransformer(os.getenv("EMBEDDING_NAME"))

def similarity_score(text):

    stored_posts = []

    if not stored_posts:
        return 0

    stored_embeddings = model.encode(stored_posts, convert_to_tensor=True)

    new_embedding = model.encode(text, convert_to_tensor=True)

    cosine_scores = util.cos_sim(new_embedding, stored_embeddings)[0]

    return cosine_scores.sum().item()

class SocialMediaMonitor:

    CATEGORY_MAP = {
        "Ransomware": ["ransomware"],
        "Data Breach": ["data breach", "data leak", "breach"],
        "Hacking": ["hacked", "hacking", "cyber attack"],
        "Malware": ["malware", "trojan", "virus"],
        "Identity Theft": ["identity theft"],
        "Phishing": ["phishing"],
        "Online Fraud": ["online fraud", "scam", "fraud"],
        "Cyber Bullying": ["cyber bullying", "harassment", "threat"]
    }

    RISK_TERMS = {
        "critical": ["massive", "millions", "nationwide", "government"],
        "high": ["bank", "financial", "hospital", "police"],
        "medium": ["user", "account", "password", "wallet"]
    }

    def __init__(self, handle, password):
        self.client = Client()
        self.client.login(handle, password)

    def _build_query(self):
        keywords = []
        for terms in self.CATEGORY_MAP.values():
            keywords.extend(terms)
        return " OR ".join([f'"{k}"' for k in keywords])

    def _classify_post(self, text):
        text = text.lower()
        for category, keywords in self.CATEGORY_MAP.items():
            if any(keyword in text for keyword in keywords):
                return category
        return "Others"

    def _risk_score(self, text):
        text = text.lower()
        score = 0

        for weight, keywords in self.RISK_TERMS.items():
            if any(keyword in text for keyword in keywords):
                if weight == "critical":
                    score += 3
                elif weight == "high":
                    score += 2
                elif weight == "medium":
                    score += 1

        return score

    def fetch_recent_incidents(self, limit=50):

        query_string = self._build_query()

        response = self.client.app.bsky.feed.search_posts(
            params={
                "q": query_string,
                "limit": limit,
                "sort": "latest"
            }
        )

        if not response.posts:
            return pd.DataFrame()

        posts_data = []

        for post in response.posts:

            if not hasattr(post.record, "text"):
                continue

            full_text = post.record.text
            category = self._classify_post(full_text)
            risk_score = self._risk_score(full_text)

            post_rkey = post.uri.split("/")[-1]

            posts_data.append({
                "source_id": post.cid,
                "platform": "Bluesky",
                "author": post.author.handle,
                "timestamp": post.record.created_at,
                "category": category,
                "risk_score": risk_score,
                "content": full_text[:80],
                "url": f"https://bsky.app/profile/{post.author.handle}/post/{post_rkey}"
            })

        df = pd.DataFrame(posts_data)

        # Optional: sort by severity
        df = df.sort_values(by="risk_score", ascending=False)

        return df
    
    def get_embedding(self, text):
        """Converts a single string into a vector (list of numbers)."""
        return self.model.encode(text, convert_to_tensor=True)

# --- EXECUTION ---
if __name__ == "__main__":

    similarity_score("hello")

    try:

        monitor = SocialMediaMonitor(BSKY_HANDLE, BSKY_PASSWORD)
        
        df = monitor.fetch_recent_incidents(limit=50)
        
        if not df.empty:
            
            print(df[['timestamp', 'author', 'content', 'is_flagged']].to_string(index=False))
            
            print(f"\n[System] Ingested {len(df)} signals. Sending to Context Engine...")
        else:
            print("No recent posts found for these topics.")

    except Exception as e:
        print(f" Error: {e}")
        print("Tip: Ensure your handle and app password are correct.")



# class SocialMediaMonitor:
#     def __init__(self, client_id, client_secret, user_agent):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.reddit = praw.Reddit(
#             client_id=client_id,
#             client_secret=client_secret,
#             user_agent=user_agent
#         )

#         self.target_subreddits = ['Scams', 'CyberSecurity', 'IdentityTheft', 'fraud']

#     def get_embedding(self, text):
#         """Converts a single string into a vector (list of numbers)."""
#         return self.model.encode(text, convert_to_tensor=True)

#     def fetch_recent_incidents(self, limit=10):
#         """
#         Fetches the 'limit' most recent posts from target subreddits.
#         """
#         print(f" Connecting to Social Stream (Reddit)...")
#         posts_data = []

#         # Combine subreddits into one stream (e.g., "Scams+fraud")
#         subreddit_query = "+".join(self.target_subreddits)
#         subreddit = self.reddit.subreddit(subreddit_query)

#         # Fetch 'New' posts to simulate real-time intake
#         for post in subreddit.new(limit=limit):
            
#             # 1. Basic Extraction
#             title = post.title
#             body = post.selftext[:500] + "..." # Truncate long text
#             full_text = f"{title} {body}"
            
#             # 2. Simple Keyword Check (The "Heuristic" Layer)
#             risk_keywords = ['urgent', 'money', 'bank', 'police', 'hack', 'steal', 'wallet']
#             flagged = any(word in full_text.lower() for word in risk_keywords)

#             # 3. Sentiment Analysis (The "Sentiment Engine")
#             # Polarity: -1 (Negative/Angry) to +1 (Positive/Happy)
#             # blob = TextBlob(full_text)
#             # sentiment_score = blob.sentiment.polarity
            
#             # 4. Construct the Data Object
#             incident = {
#                 'source_id': post.id,
#                 'platform': 'Reddit',
#                 'subreddit': post.subreddit.display_name,
#                 'timestamp': datetime.datetime.fromtimestamp(post.created_utc),
#                 'content': title, # Using title as the main headline
#                 'full_text': full_text,
#                 # 'sentiment_score': round(sentiment_score, 2),
#                 'is_flagged': flagged,
#                 'url': post.url
#             }
#             posts_data.append(incident)

#         return pd.DataFrame(posts_data)

# # text api -> swa

# # --- CONFIGURATION (REPLACE WITH YOUR KEYS) ---
# CLIENT_ID = 'YOUR_REDDIT_CLIENT_ID'
# CLIENT_SECRET = 'YOUR_REDDIT_CLIENT_SECRET'
# USER_AGENT = 'RiskBot'

# # --- EXECUTION ---
# if __name__ == "__main__":
#     try:
#         # 1. Initialize Monitor
#         monitor = SocialMediaMonitor(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
#         # 2. Get Data
#         df = monitor.fetch_recent_incidents(limit=5)
        
#         # 3. Display Result (Mocking the "Ingestion" phase)
#         print("\n--- 🚨 RECENT DETECTED SIGNALS ---")
#         if not df.empty:
#             # Show only high-risk or relevant columns
#             print(df[['timestamp', 'subreddit', 'content', 'is_flagged']].to_string(index=False))
            
#             # Example of how you'd pass this to your Risk Model
#             print(f"\n[System] Ingested {len(df)} signals. Sending to Context Engine...")
#         else:
#             print("No recent posts found.")

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         print("Tip: Did you replace the CLIENT_ID and SECRET with real keys?")