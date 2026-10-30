import os
from dotenv import load_dotenv
import requests

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from rag import vector_store

load_dotenv()

# Tool to give out AI Agent capability to search through the internet
web_search_tool = DuckDuckGoSearchRun()

# # Tool to get job recommendations
@tool
def get_job_recommendation(what: str, salary_min: int) -> str:
    """
    Searches live Indian job listings (Adzuna) by keyword and minimum salary.

    Input Parameters:
    what (str) - One or more SINGLE KEYWORDS, space-separated. This is NOT a job
        title phrase — each space-separated word is matched as a separate keyword,
        not as part of one title. Use 1-2 core keywords, not a full title.
        Good: "python", "datascientist", "python fastapi", "machinelearning".
        Avoid: "machine learning engineer" (three separate required keywords,
        which over-narrows results since a listing must contain all three).
        Never pass a parameter name (like "salary_min") or an instruction as this value.
    salary_min (int) - Minimum ANNUAL salary in Indian Rupees (INR), e.g. 400000 for ₹4 LPA.
        Not USD. Typical entry-level India tech salaries start around 300000-600000.

    Returns:
    JSON with a "results" list. Each job object includes: title, company.display_name,
    description, salary_min, salary_max (all INR), location.display_name, and
    redirect_url (the direct apply link for that job).

    Example: get_job_recommendation(what="python", salary_min=400000)
    Example: get_job_recommendation(what="datascientist", salary_min=500000)
    """
    url = f'https://api.adzuna.com/v1/api/jobs/in/search/1?app_id={os.getenv("ADZUNA_APP_ID")}&app_key={os.getenv("ADZUNA_API_KEY")}&what={what}&salary_min={salary_min}'

    response = requests.get(url)

    return response.json()

@tool
def get_resume_data(query:str) -> str:
    """
    Retrieves chunks of the candidate's resume via similarity search.

    Input Parameters:
    query (str) - A resume TOPIC to search for, e.g. "skills", "work experience",
        "education", "years of experience". This must be a topic that appears IN
        the resume itself — not an instruction, judgment, or classification task
        (e.g. do NOT query "seniority level classification"; instead query "work
        experience" and determine seniority yourself from what's returned).

    Returns:
    A list of matching resume text chunks. Content is chunked, so multiple calls
    with different topic queries are usually needed to get a full picture.

    Example: get_resume_data(query="skills")
    """

    response = vector_store.similarity_search(query) # This will return the similarity search results. Mention to the student this is different than the one used yesterday. 
                                                     # Show the difference
    return response

if __name__ == "__main__":
    response = web_search_tool.invoke("LA Olympics 2028") # First try this
    print(response)

    #print(get_job_recommendation("Data Analyst", 300000)) # Run this without tool decorator then with the decorator. It will give an error the second time
    print(get_job_recommendation.invoke({"what": "Data Analyst", "salary_min": 300000})) # This will work with tool decorator

    print(f"\nTool Name: {get_job_recommendation.name}")
    print(f"\nTool Arguments: {get_job_recommendation.args}")
    print(f"\nTool Description: {get_job_recommendation.description}")

    # print(get_resume_data.invoke({"query": "Skills"}))