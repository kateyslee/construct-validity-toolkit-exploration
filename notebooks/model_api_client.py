import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import json
import ast
import re

from litellm import completion
from mistralai import Mistral
from openai import OpenAI
import anthropic

# from config import keys


class ModelAPIClient:
    """
    Client for making API calls to various language model providers.
    Handles validation and conversion of responses.
    """
    @staticmethod
    def reason_convert(input_string : str, low_score : int =0, high_score : int=1) -> Tuple[int, str]:
        print("Input String: ", input_string)
        score_match = re.search(r'Score:\s*(\d)', input_string)
        # print(score_match)
        explanation_match = re.search(r'Explanation:\s*(.+)', input_string, re.DOTALL)

        score = int(score_match.group(1)) if score_match else -1
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found"
        
        return score, explanation
    
        # splits = input_string.split(maxsplit=1)
        # if len(splits) <= 1:
        #     return (-1, "")
        # score, reason = splits[0].strip().strip("'").strip('"'), splits[1]
        # try:
        #     score = int(score)
        #     # Check if it's in the range 1-100
        #     if low_score <= score <= high_score:
        #         return (score, reason)
        #     else:
        #         return (-1, "")
        # except ValueError:
        #     # Not a valid integer
        #     return (-1, "")

    
    @staticmethod
    def validate_and_convert(input_string: str) -> int:
        """
        Validate and convert a string response to a numeric rating between 1-100.
        
        Args:
            input_string: Response string from the model
            
        Returns:
            int: Converted rating (1-100) or -1 if invalid
        """
        # Remove whitespace and check if the string is a valid number format
        cleaned_string = input_string.strip().strip("'").strip('"')
        
        # Check if it's a valid integer
        try:
            num = int(cleaned_string)
            # Check if it's in the range 1-100
            if 0 <= num <= 100:
                return num
            else:
                return -1
        except ValueError:
            # Not a valid integer
            return -1
    
    @staticmethod
    def call_api(
        prompt: str, 
        provider: str, 
        model: str, 
        max_tokens: int = 5, 
        n: int = 1,
        mock: bool = False
    ) -> Union[str, List[str]]:
        """
        Call the specified API provider to generate responses.
        
        Args:
            prompt: Input prompt
            provider: API provider ('openai', 'mistral', 'llama', or 'anthropic')
            model: Model name
            max_tokens: Maximum tokens in response
            n: Number of responses to generate
            mock: If True, return a random number between 0 and 100 as a string
            
        Returns:
            str or list: Generated response (or empty list if error)
        """
        
        # If mock is enabled, return a random number
        if mock:
            random_rating = str(np.random.randint(0, 101))  # 0 to 100 inclusive
            return f"{random_rating} BLAH"


        messages = [{"role": "user", "content": prompt}]
        
        # Add small delay to avoid rate limits
        time.sleep(2)
        
        try:
            if provider == 'openai':
                chat_response = completion(
                    model=f"{provider}/{model}",
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n
                )
                return chat_response.choices[0].message.content

            elif provider == 'mistral':
                client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=0.7
                )
                return chat_response.choices[0].message.content

            elif provider == 'llama':
                client = OpenAI(
                    api_key=os.environ["LLMANA_API_KEY"],
                    base_url="https://api.llama.com/compat/v1/"
                )
                return client.chat.completions.create(
                    max_tokens=max_tokens,
                    model=model,
                    messages=messages,
                ).choices[0].message.content

            elif provider == 'anthropic':
                client = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                )
                chat_response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                return chat_response.content[0].text

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"Error calling {provider} API: {str(e)}")
            return []
