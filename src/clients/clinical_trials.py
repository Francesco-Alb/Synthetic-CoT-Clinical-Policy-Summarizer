
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass, asdict, field

@dataclass
class ClientConfiguration:
    base_url: str = "https://clinicaltrials.gov/api/v2/studies"
    retries: int = 3
    min_delay: float = 1.2
    headers: dict = field(default_factory=lambda: {
        "User-Agent": "ClinicalTrialsProject/1.0",
        "Accept": "application/json"
    })

@dataclass
class SearchConfig:
    search_expr: list[str] = field(default_factory=lambda: ["Diabetes"])
    overall_status: str = "COMPLETED"
    fields: list[str] = field(default_factory=lambda: ["NCTId", "BriefTitle", "OverallStatus", "BriefSummary", "DetailedDescription", "Condition"])
    total_to_collect: int = 2000
    page_size: int = 500

@dataclass
# NOTE: Use default_factory for mutable default values to prevent unintended shared state.
class CFG:
    client: ClientConfiguration = field(default_factory=ClientConfiguration)
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self):
        return asdict(self)

class ClinicalTrialsApiClient:
    """
    Minimal, polite HTTP client for the ClinicalTrialsApi.

    Design goals:
    - Centralize HTTP concerns (timeouts, retries, rate control).
    - Avoid accidental API overload or client-side bans.
    - Remain simple enough for single-threaded, educational usage.

    This client deliberately trades maximal throughput for robustness
    and clarity of behavior.
    """
    def __init__(self, client_config: ClientConfiguration):
        """
        Initialize the API client and configure HTTP-level behavior.

        Parameters
        ----------
        client_config : ClientConfiguration
            Configuration object for the API client.
        """

        self.base_url = client_config.base_url
        self.min_delay = client_config.min_delay
        self._last_request_ts = 0.0

        self.session = requests.Session()
        self.session.headers.update(client_config.headers)

        # Retry strategy
        retry_strat = Retry(
            total=client_config.retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
            respect_retry_after_header=True
        )

        adapter = HTTPAdapter(max_retries=retry_strat)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _polite_get(self, url, params, timeout=(3.05, 10)):
        """
        Perform a GET request while enforcing client-wide politeness rules.

        This method is the *single choke point* for all outbound requests.
        If you later need logging, metrics, authentication refresh, or
        adaptive throttling, this is where it belongs.

        Parameters
        ----------
        url : str
            Fully-qualified request URL.
        params : dict
            params forwarded to `requests.Session.get`.

        Returns
        -------
        requests.Response
            A successful HTTP response (status >= 200 and < 400).

        Raises
        -------
        requests.exceptions.HTTPError
            Raised after retries are exhausted and the response is still
            an error. Callers are expected to decide whether to stop or
            degrade gracefully.
        """
        # Enforce minimum spacing between requests.
        # This ensures a minimum `self.min_delay` seconds pass *between* the completion of one request
        # and the initiation of the next, preventing unnecessary extra waits if the previous request
        # already took a long time, making it more efficient than a simple `time.sleep` before every request.
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay:
            waiting_time = self.min_delay - elapsed
            time.sleep(waiting_time)
            print(f"Sleeping for {waiting_time:.2f} seconds.")

        response = self.session.get(url, params=params)

        self._last_request_ts = time.time()
        response.raise_for_status()
        return response

    def get_studies(self, search_config: SearchConfig, search_mode: str = "OR"):
      """
      Fetch studies from the API.

      Parameters
      ----------
      search_config : SearchConfig
          Configuration object containing search parameters.
      search_mode : str, optional
          Determines how multiple search_expr terms are handled.
          "OR" (default): Joins terms with " OR " in a single API call.
          "INDIVIDUAL": Performs separate API calls for each term and combines results.

      Returns
      -------
      list of dict
          A list of dictionaries, where each dictionary represents a study
          and contains the requested fields.
      """

      collected = []
      collected_nct_ids = set() # To store NCTIds and prevent duplicates

      print(
          f"Fetching studies.\n"
          f"Target: {search_config.total_to_collect} | Page size: {search_config.page_size}"
          f" | Search Mode: {search_mode}"
          )

      if search_mode == "OR":
        # Combine all search expressions into a single query term with " OR "
        query_terms_to_process = [" OR ".join(search_config.search_expr)]
        term_targets = [search_config.total_to_collect] # Target all for single query
      elif search_mode == "INDIVIDUAL":
        query_terms_to_process = search_config.search_expr
        num_terms = len(query_terms_to_process)
        base_target = search_config.total_to_collect // num_terms
        remainder = search_config.total_to_collect % num_terms
        # Distribute the remainder evenly among the first 'remainder' terms
        term_targets = [base_target + 1 if i < remainder else base_target for i in range(num_terms)]
      else:
        raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'OR' or 'INDIVIDUAL'.")

      # Iterate over the determined query terms with their respective targets
      for i, current_query_term in enumerate(query_terms_to_process):
        if len(collected) >= search_config.total_to_collect: # Overall limit check
            break

        current_term_target = term_targets[i]
        # print a log stating what term is it looking for now
        print(f"Collecting for term: {current_query_term}")
        collected_for_current_term_count = 0

        page_token = None

        # Loop to fetch pages for the current query term
        while (len(collected) < search_config.total_to_collect and
               collected_for_current_term_count < current_term_target):
          params = {
              "query.term": current_query_term,
              "filter.overallStatus": search_config.overall_status,
              "sort": "@relevance",
              "pageSize": search_config.page_size,
              "fields": ",".join(search_config.fields),
              "format": "json",
              }

          if page_token is not None:
            params["pageToken"] = page_token

          try:
            response = self._polite_get(
                self.base_url, params=params
                )
            data = response.json()

            studies_on_page = data.get("studies", [])
            if not studies_on_page: # No studies on this page, or end of results for this term
                break

            for study in studies_on_page:
                if len(collected) >= search_config.total_to_collect:
                    break # Stop if overall total is met

                nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if nct_id and nct_id not in collected_nct_ids:
                    modified_study = study.copy() # Make a copy
                    modified_study['search_term_used'] = current_query_term # Add the search term
                    collected.append(modified_study)
                    collected_nct_ids.add(nct_id)
                    collected_for_current_term_count += 1

                if collected_for_current_term_count >= current_term_target:
                    break # Stop if target for current term is met

            # Check if all studies for the current term have been collected or if total_to_collect is reached
            page_token = data.get("nextPageToken")
            if not page_token: # No more pages for this term
                break

          except requests.exceptions.HTTPError as e:
              print(f"HTTP Error occurred for term '{current_query_term}': {e}")
              break # Stop fetching for current term
          except Exception as e:
              print(f"An unexpected error occurred for term '{current_query_term}': {e}")
              break # Stop fetching for current term

      # Ensure that no more than total_to_collect items are returned (in case of over-collection due to page_size)
      collected = collected[:search_config.total_to_collect]
      print(f"Total items collected via pagination: {len(collected)}")
      return collected
