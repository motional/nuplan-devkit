import logging
import os
from typing import Any, Dict, Optional

import requests

URLS: Dict[str, str] = {
    "update_submission": "/api/jobs/challenge/{}/update_submission/",
}

logger = logging.getLogger(__name__)


class EvalaiInterface:
    """Interface to use EvalAI APIs."""

    def __init__(self, api_server: str = "https://eval.ai") -> None:
        """
        :param api_server: The URL of the api server.
        """
        self.CHALLENGE_PK = os.getenv('EVALAI_CHALLENGE_PK')
        self.EVALAI_AUTH_TOKEN = os.getenv('EVALAI_PERSONAL_AUTH_TOKEN')
        assert self.CHALLENGE_PK, "Missing required environmental variable EVALAI_CHALLENGE_PK!"
        assert self.EVALAI_AUTH_TOKEN, "Missing required environmental variable EVALAI_PERSONAL_AUTH_TOKEN!"

        self.EVALAI_API_SERVER = api_server

    def update_submission_data(self, data: Dict[str, str]) -> Any:
        """
        Updates the status of a submission according to the input data.
        :param data: The information to update the submission. The submission is specified in data.
        :return: Server response.
        """
        url = self._format_url("update_submission")
        response = self._request(url, "PUT", data)
        return response

    def _request(self, url: str, method: str, data: Optional[Dict[str, str]] = None) -> str:
        """
        Creates request according to parameters.
        :param url: Target url
        :param method: Method (i.e. 'PUT')
        :param data: Optional payload
        :return: Response from server.
        :raises RequestException: If connection fails.
        """
        try:
            response = requests.request(method=method, url=url, headers=self._get_request_headers(), data=data)
            response.raise_for_status()
            logger.info(response.json())
        except requests.exceptions.RequestException as e:
            logger.error("Could not establish connection with EvalAI server at %s" % self.EVALAI_API_SERVER)
            logger.error(e)
            raise e
        return response.json()  # type: ignore

    def _format_url(self, api: str) -> str:
        """
        Creates correct URL using api and server.
        :param api: The requested API.
        :return: The formatted URL.
        """
        assert api in URLS, f"Requested API unavailable, available ones are {URLS}"
        return f'{self.EVALAI_API_SERVER}{URLS.get(api).format(self.CHALLENGE_PK)}'  # type: ignore

    def _get_request_headers(self) -> Dict[str, str]:
        """
        Creates correct headers for authentication in requests.
        :return: The header with the authentication token.
        """
        return {"Authorization": f"Bearer {self.EVALAI_AUTH_TOKEN}"}
