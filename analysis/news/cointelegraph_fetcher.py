import requests

def fetch_cointelegraph_news():
    url = "https://cointelegraph.com/api/v1/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
        "application/json"
    ):
        return resp.json()
    else:
        print(
            f"Erreur Cointelegraph: code={resp.status_code}, content-type={resp.headers.get('Content-Type')}"
        )
        return []
