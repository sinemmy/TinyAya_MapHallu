import cohere
from dotenv import load_dotenv
import os
import json
import time
from uuid import uuid4

# Load .env file
load_dotenv()

# Access variables
COHERE_API = os.getenv("COHERE_API")

import cohere
co = cohere.ClientV2("YOUR_API_KEY")

DEBUG_LOG_PATH = "/Users/shubham/Documents/F/Research_projects/TinyAya_MapHalluc/.cursor/debug-8483aa.log"
DEBUG_SESSION_ID = "8483aa"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{uuid4().hex[:8]}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

def query_model(query:str, model: str ="tiny-aya-global", temp:float = 0.3, logprobs:bool = False):
    '''Function to query the model with a given query, model name, temperature, and logprobs setting. logprobs is set to False by default.
    Returns the full response from the model. To get only the text response, use the get_text_from_response function. To get the logprobs, use the get_logprobs_from_response function.
    '''
    co = cohere.ClientV2(api_key=COHERE_API)
   
    response = co.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ],
        response_format={
          "type": "json_object"
        },
        temperature= temp,
        model=model,
        logprobs=logprobs
        
    )
    return response

def get_text_from_response(response):
    '''Function to extract the text response from the full response returned by the query_model function.'''
    return response.message.content[0].text

def get_logprobs_from_response(response):
    '''Function to extract the logprobs from the full response returned by the query_model function.'''
    # region agent log
    _debug_log(
        run_id="baseline",
        hypothesis_id="H3",
        location="helpers.py:get_logprobs_from_response:entry",
        message="Inspect response logprobs availability",
        data={
            "response_type": type(response).__name__,
            "has_logprobs": hasattr(response, "logprobs"),
        },
    )
    # endregion
    if hasattr(response, "logprobs"):
        log_probabilities = response.logprobs
        # region agent log
        _debug_log(
            run_id="baseline",
            hypothesis_id="H2",
            location="helpers.py:get_logprobs_from_response:has_logprobs",
            message="Inspect raw logprobs container",
            data={
                "container_type": type(log_probabilities).__name__,
                "container_len": len(log_probabilities) if hasattr(log_probabilities, "__len__") else None,
                "first_item_type": type(log_probabilities[0]).__name__
                if hasattr(log_probabilities, "__len__") and len(log_probabilities) > 0
                else None,
                "first_item_attrs": sorted([a for a in dir(log_probabilities[0]) if not a.startswith("_")])[:20]
                if hasattr(log_probabilities, "__len__") and len(log_probabilities) > 0
                else [],
            },
        )
        # endregion
        return log_probabilities
    else:
        print('No logprobs found in response.')
        # region agent log
        _debug_log(
            run_id="baseline",
            hypothesis_id="H4",
            location="helpers.py:get_logprobs_from_response:no_logprobs",
            message="Response missing logprobs attribute",
            data={"response_type": type(response).__name__},
        )
        # endregion
        return None