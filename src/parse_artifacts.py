import json

issues_file = "../data/raw/issues.json"
comments_file = "../data/raw/comments.json"
output_file = "../data/processed/artifacts.json"

with open(issues_file, "r") as f:
    issues = json.load(f)

with open(comments_file, "r") as f:
    comments = json.load(f)

# Build lookup tables
issue_id_to_number = {i["id"]: i["number"] for i in issues}
issue_id_to_state = {i["id"]: i["state"] for i in issues}
issue_id_to_title = {i["id"]: i["title"] for i in issues}

artifacts = []

# Convert issues → artifacts
for issue in issues:

    text = issue.get("body", "")

    artifacts.append({
        "artifact_id": issue["id"],
        "artifact_type": "issue",
        "issue_number": issue["number"],
        "issue_title": issue.get("title"),
        "issue_state": issue.get("state"),
        "source_issue_id": issue["id"],
        "author": issue.get("author"),
        "timestamp": issue.get("created_at"),
        "text": text
    })

# Convert comments → artifacts
for comment in comments:

    artifacts.append({
        "artifact_id": comment["id"],
        "artifact_type": "comment",
        "issue_number": issue_id_to_number.get(comment["issue_id"]),
        "issue_title": issue_id_to_title.get(comment["issue_id"]),
        "issue_state": issue_id_to_state.get(comment["issue_id"]),
        "source_issue_id": comment["issue_id"],
        "author": comment.get("author"),
        "timestamp": comment.get("created_at"),
        "text": comment.get("body", "")
    })

with open(output_file, "w") as f:
    json.dump(artifacts, f, indent=2)

print(f"{len(artifacts)} artifacts written to {output_file}")