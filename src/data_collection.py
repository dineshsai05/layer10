import os
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["GITHUB_TOKEN"]

url = "https://api.github.com/graphql"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

query = """
query($cursor:String){
  repository(owner:"langchain-ai", name:"langchain"){
    issues(first:50, after:$cursor, states:[OPEN,CLOSED]){
      pageInfo{
        hasNextPage
        endCursor
      }
      nodes{
        id
        number
        title
        body
        createdAt
        state
        author{login}

        comments(first:100){
          nodes{
            id
            body
            createdAt
            author{login}
          }
        }
      }
    }
  }
}
"""

cursor = None
has_next = True

issues = []
comments = []

print("Collecting issues via GraphQL...")

while has_next:

    response = requests.post(
        url,
        headers=headers,
        json={"query": query, "variables": {"cursor": cursor}}
    )

    data = response.json()

    if "errors" in data:
        print(data)
        break

    issue_nodes = data["data"]["repository"]["issues"]["nodes"]

    for issue in issue_nodes:

        issues.append({
            "id": issue["id"],
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
            "state": issue["state"],
            "created_at": issue["createdAt"],
            "author": issue["author"]["login"] if issue["author"] else None
        })

        for c in issue["comments"]["nodes"]:
            comments.append({
                "id": c["id"],
                "issue_id": issue["id"],
                "body": c["body"],
                "author": c["author"]["login"] if c["author"] else None,
                "created_at": c["createdAt"]
            })

    page = data["data"]["repository"]["issues"]["pageInfo"]

    cursor = page["endCursor"]
    has_next = page["hasNextPage"]

    print(f"Issues: {len(issues)} | Comments: {len(comments)}")

print("Saving data...")

with open("issues_graphql.json","w") as f:
    json.dump(issues,f)

with open("comments_graphql.json","w") as f:
    json.dump(comments,f)

print("Done")
