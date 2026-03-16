import cohere
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
COHERE_API = os.getenv("COHERE_API")

import cohere
co = cohere.ClientV2("YOUR_API_KEY")

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
    if hasattr(response, "logprobs"):
        log_probabilities = response.logprobs
        return log_probabilities
    else:
        print('No logprobs found in response.')
        return None