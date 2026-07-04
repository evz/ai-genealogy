import logging

import requests

logger = logging.getLogger(__name__)


class MediaWikiClient:
    """Client for MediaWiki API operations."""

    def __init__(self, api_url: str, username: str, password: str):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GenealogyWikiBot/1.0"})
        self._login(username, password)

    def _get_token(self, token_type: str = "csrf") -> str:
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "meta": "tokens", "type": token_type, "format": "json"},
        )
        resp.raise_for_status()
        return resp.json()["query"]["tokens"][f"{token_type}token"]

    def _login(self, username: str, password: str) -> None:
        login_token = self._get_token("login")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "login",
                "lgname": username,
                "lgpassword": password,
                "lgtoken": login_token,
                "format": "json",
            },
        )
        resp.raise_for_status()
        result = resp.json()
        if result["login"]["result"] != "Success":
            raise ValueError(f"MediaWiki login failed: {result['login']['result']}")
        logger.info("Logged in to MediaWiki as %s", username)

    def create_or_update_page(self, title: str, content: str, summary: str = "Auto-generated") -> dict:
        token = self._get_token("csrf")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "edit",
                "title": title,
                "text": content,
                "summary": summary,
                "token": token,
                "format": "json",
                "bot": "1",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def page_exists(self, title: str) -> bool:
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "titles": title, "format": "json"},
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        return "-1" not in pages

    def create_template(self, name: str, content: str) -> dict:
        return self.create_or_update_page(f"Template:{name}", content, summary="Template setup")

    def create_category_page(self, name: str, description: str) -> dict:
        return self.create_or_update_page(f"Category:{name}", description, summary="Category setup")
