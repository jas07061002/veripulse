import logging
import pdb
import os
import sys
import requests
import re
import urllib.parse
import torch
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from bs4 import BeautifulSoup
from pymongo import MongoClient
from dotenv import load_dotenv


# Set up logging to stream to stdout
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger()

# Initialize FastAPI app
app = FastAPI()
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
headers = {"Authorization": "Bearer {HUGGINGFACE_API_KEY}"}


def get_huggingface_api_response(user_query: str):
    try:
        # Send request to Hugging Face API
        response = requests.post(API_URL, headers=headers, json={"inputs": user_query})
        
        # Check if the response is successful
        if response.status_code == 200:
            # Extract the response JSON
            response_data = response.json()
            
            # Log the entire response to inspect its structure
            logger.info(f"Full response data: {response_data}")
            
            # Ensure 'generated_text' exists in the response
            if isinstance(response_data, list) and len(response_data) > 0 and 'generated_text' in response_data[0]:
                raw_text = response_data[0]['generated_text']
                
                # Clean up the generated text to remove any unwanted special characters or strange formatting
                clean_text = re.sub(r'[^\x00-\x7F]+', '', raw_text)  # Removes non-ASCII characters
                clean_text = re.sub(r'[(){}[\]]', '', clean_text)  # Removes parentheses, braces, and brackets
                clean_text = re.sub(r'\n+', ' ', clean_text)  # Replace multiple newlines with a single space
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Remove extra spaces
                return clean_text
            else:
                raise Exception(f"'generated_text' not found in the response: {response_data}")
        
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    
    except Exception as e:
        # Log the exception with more context to understand where it failed
        logger.error(f"Error in Hugging Face API response: {e}")
        raise HTTPException(status_code=500, detail=f"Error with Hugging Face API: {e}")

def get_response(user_message):
    try:
        logger.info(f"Received message: {user_message}")
        response = get_huggingface_api_response(user_message)
        logger.info(f"Response from Hugging Face: {response}")
        return {"search_results": response}
    except Exception as e:
        # Log the exception with additional details for debugging
        logger.error(f"Error while processing the response: {e}")
        raise HTTPException(status_code=500, detail=f"Error while processing response: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Backend!"}

@app.get("/generate_response/")
def generate_response(query: str):
    try:
        response = get_response(query)  # Pass the query to the response function
        return response # Adjusted for correct response parsing
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {e}")

@app.get("/fetch_source/")
def get_source(url: str):
    return search_for_sources(url)

def search_for_sources(query: str):
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        # Loop through the links and handle relative URLs
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle valid search result links
            if href.startswith('https://'):
                results.append(href)
            elif href.startswith('/l/?kh='):
                full_url = f"https://duckduckgo.com{href}"
                results.append(full_url)

        # Remove duplicates using a set
        unique_results = list(set(results))

        # If no valid links are found, return a message
        if not unique_results:
            return {"search_results": "No valid results found for the query."}

        return {"search_results": unique_results}
    else:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Error from DuckDuckGo: {response.text}"
        )

def summarize_content(content: str):
    try:
        # Use OpenAI to summarize the content (you could also use other models like BERT or a custom model)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": f"Summarize this: {content}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing content: {e}")

def summarize_all_sources(query: str):
    openai_answer, training_cutoff = get_openai_response(query)
    
    # Summarize OpenAI response
    openai_summary = summarize_content(openai_answer)
    
    # Fetch sources and summarize each of them
    search_results = search_for_sources(query)["search_results"]
    source_summaries = []
    
    for url in search_results:
        try:
            # Scrape content from each URL (could be done with a scraping function)
            page_content = fetch_page_content(url)
            source_summaries.append(summarize_content(page_content))
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
    
    return {
        "openai_answer": openai_answer,
        "openai_summary": openai_summary,
        "source_summaries": source_summaries
    }

def fact_check(query: str):
    try:
        response = requests.get(f"https://api.factcheck.org/check?query={query}")
        data = response.json()
        return data  # Return fact-checked data or validation status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with fact-checking API: {e}")

@app.get("/fact_check/")
def fact_check_response(query: str):
    return fact_check(query)


