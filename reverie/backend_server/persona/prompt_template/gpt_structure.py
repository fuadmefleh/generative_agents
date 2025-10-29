"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Ollama APIs.
"""
import json
import random
import time
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import requests
from pydantic import BaseModel, Field, validator, ValidationError



from utils import *

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()

  response = requests.post(
    'http://localhost:11434/api/generate',
    json={
      'model': 'llama3.1:8b',
      'prompt': prompt,
      'stream': False
    }
  )
  response.raise_for_status()
  return response.json()['response']


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to Ollama
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    response = requests.post(
      'http://localhost:11434/api/generate',
      json={
        'model': 'llama3.1:8b',
        'prompt': prompt,
        'stream': False
      }
    )
    response.raise_for_status()
    return response.json()['response']
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to Ollama
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    response = requests.post(
      'http://localhost:11434/api/generate',
      json={
        'model': 'llama3.1:8b',
        'prompt': prompt,
        'stream': False
      }
    )
    response.raise_for_status()
    return response.json()['response']
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request_structured(
    prompt: str, 
    gpt_parameters,
    response_model: Optional[BaseModel] = None
) -> Dict[str, Any]:
    """
    Make a structured request to Ollama with optional Pydantic model validation.
    
    Args:
        prompt: The prompt string
        gpt_parameters: GPTParameters model with request parameters
        response_model: Optional Pydantic model class for structured output
    
    Returns:
        Dictionary containing response and optional structured data
    """
    temp_sleep()
    
    try:
        # Build request payload
        payload = {
            'model': gpt_parameters["model"],
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': gpt_parameters["temperature"],
                'num_predict': gpt_parameters["max_tokens"],
                'top_p': gpt_parameters["top_p"],
            }
        }
        
        # Add stop tokens if provided
        if gpt_parameters["stop"]:
            payload['options']['stop'] = gpt_parameters["stop"]

        # Add format instruction if using structured output
        if response_model:
            # Add JSON schema instruction to prompt
            schema = response_model.schema()
            structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

JSON Response:"""
            
            print( "STRUCTURED PROMPT:::: ", structured_prompt)
            payload['prompt'] = structured_prompt
            payload['format'] = 'json'
        
        # Make request
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        raw_response = result.get('response', '')
        
        # Parse structured output if model provided
        if response_model:
            try:
                # Try to parse JSON from response
                json_str = raw_response.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                
                json_data = json.loads(json_str)
                structured_data = response_model(**json_data)
                
                return {
                    'success': True,
                    'raw_response': raw_response,
                    'structured_data': structured_data,
                    'data': structured_data.dict()
                }
            except (json.JSONDecodeError, ValidationError) as e:
                return {
                    'success': False,
                    'raw_response': raw_response,
                    'error': str(e),
                    'structured_data': None
                }
        
        return {
            'success': True,
            'raw_response': raw_response,
            'structured_data': None
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'raw_response': None,
            'error': f"Request failed: {str(e)}",
            'structured_data': None
        }
    except Exception as e:
        return {
            'success': False,
            'raw_response': None,
            'error': f"Unexpected error: {str(e)}",
            'structured_data': None
        }


def safe_generate_structured_response(
    prompt: str,
    gpt_parameters,
    response_model: BaseModel,
    repeat: int = 5,
    fail_safe_response: Any = None,
    verbose: bool = False
) -> Any:
    """
    Safely generate a structured response with retries.
    
    Args:
        prompt: The prompt string
        gpt_parameters: GPTParameters model
        response_model: Pydantic model for response validation
        repeat: Number of retry attempts
        fail_safe_response: Default response if all attempts fail
        verbose: Print debug information
    
    Returns:
        Structured response or fail-safe value
    """
    if verbose:
        print(f"Prompt: {prompt[:200]}...")
    
    for i in range(repeat):
        result = GPT_request_structured(prompt, gpt_parameters, response_model)
        
        if result['success'] and result['structured_data']:
            if verbose:
                print(f"Success on attempt {i+1}")
            return result['structured_data']
        
        if verbose:
            print(f"Attempt {i+1} failed: {result.get('error', 'Unknown error')}")
            if result.get('raw_response'):
                print(f"Raw response: {result['raw_response']}")
    
    if verbose:
        print(f"All attempts failed, returning fail-safe: {fail_safe_response}")
    
    return fail_safe_response

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to Ollama
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    # Note: Ollama doesn't support all OpenAI parameters, using the most relevant ones
    response = requests.post(
      'http://localhost:11434/api/generate',
      json={
        'model': 'llama3.1:8b',
        'prompt': prompt,
        'stream': False,
        'options': {
          'temperature': gpt_parameter.get("temperature", 0.7),
          'num_predict': gpt_parameter.get("max_tokens", 50),
          'top_p': gpt_parameter.get("top_p", 1),
          'stop': gpt_parameter.get("stop", [])
        }
      }
    )
    response.raise_for_status()
    return response.json()['response']
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="llama3.1:8b"):
  """
  Get text embeddings using Ollama's embedding endpoint.
  Note: Ollama uses a different endpoint for embeddings.
  """
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  
  response = requests.post(
    'http://localhost:11434/api/embeddings',
    json={
      'model': model,
      'prompt': text
    }
  )
  response.raise_for_status()
  return response.json()['embedding']


if __name__ == '__main__':
  gpt_parameter = {"engine": "llama3.1:8b", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)