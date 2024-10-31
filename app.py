# Streamlit app

import os
from crewai import Crew, Process, Agent, Task
from crewai_tools import PDFSearchTool
import streamlit as st
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import nltk
import PyPDF2
import torch
import plotly.express as px

# Set the OpenAI API key as an environment variable
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with actual key or leave as env variable
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OpenAI.api_key = st.secrets["OPENAI_API_KEY"]

model_ID = 'gpt-4o'
os.environ["OPENAI_MODEL_NAME"] = model_ID

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# Function to extract text from the uploaded PDF resume
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text


# Function to find the missing skills for each job using CrewAI agents
def find_missing_skills_for_job(uploaded_file, job_skills):

    pdf_search_tool = PDFSearchTool(uploaded_file.name)

    job_skills_list = job_skills.strip().split(',')
    job_skills_list = [skill.strip() for skill in job_skills_list if skill.strip()]

    skill_extractor_agent = Agent(
        role='Skill Extractor',
        goal='Extract relevant skills from the provided resume',
        verbose=True,
        memory=True,
        backstory="Expert in extracting skills from resumes. Refers to the user as an unemployed student looking for opportunities.",
        tools=[pdf_search_tool],
        allow_delegation=True
    )


    # Define the missing skills agent
    missing_skills_agent = Agent(
        role='Missing Skills Finder',
        goal="Identify missing skills based on semantic similarity between the user's skills obtained from the skill_extractor_agent and job requirements {topic}.",
        verbose=True,
        memory=True,
        backstory="Expert in extracting matching skills based on semantic similarity between skills present between a candidate and the job requirements.",
        allow_delegation=False
    )

    skill_extraction_task = Task(
        description="Identify and extract key skills from the resume.",
        expected_output="A list of skills extracted from the resume.",
        agent=skill_extractor_agent,
        tools=[pdf_search_tool]
    )

    # Define the task for finding missing skills
    missing_skills_task = Task(
        description="Compare the user's skills obtained from the skill_extractor_agent with the job requirements which are {topic} and identify only 4 or lesser missing skills.",
        expected_output="A list of only 4 or lesser missing skills.",
        agent=missing_skills_agent,
    )



    my_crew = Crew(
        agents=[skill_extractor_agent, missing_skills_agent],
        tasks=[skill_extraction_task, missing_skills_task],
        verbose=False,
        process=Process.sequential,
        memory=True,
        cache=True,
        max_rpm=500,
        share_crew=True
    )

    # Kick off the agent with user skills and job skills
    result = my_crew.kickoff(inputs={'topic': job_skills_list})

    # Assume that the agent returns missing skills as a string of skills
    missing_skills = result.raw.strip().split(',')
    missing_skills = [skill.strip() for skill in missing_skills if skill.strip()]

    return missing_skills



# Function to process the uploaded resume with CrewAI agents
def process_resume_with_agents(uploaded_file, search_query):
    # Initialize PDF search tool and extract text
    pdf_search_tool = PDFSearchTool(uploaded_file.name)
    text = extract_text_from_pdf(uploaded_file)

    # Define the Agents and Tasks
    job_researcher = Agent(
      role='Job Researcher',
      goal='Search for relevant information on the topic {topic} from the provided Resume',
      verbose=True,
      memory=True,
      backstory="Expert in finding and analyzing relevant content from Resume's. Refers to the person who uploaded the resume as a sad unemployed student",
      tools=[pdf_search_tool],
      allow_delegation=True
    )


    research_task = Task(
      description="Identify and analyze {topic} specifically from the resume provided by the Job researcher. Give top 2 possible jobs, just job names.",
      expected_output="A complete word by word report on {topic}.",
      agent=job_researcher,
      tools=[pdf_search_tool]
    )

    # Construct the topic of interest based on user input
    if search_query:
        topic_of_interest = f"What kinds of jobs is the resume suited for? Also, search for '{search_query}'."
    else:
        topic_of_interest = "What kinds of jobs is the resume suited for?"

    my_crew = Crew(
    agents=[job_researcher],
    tasks=[research_task],
    verbose=False,
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=500,
    share_crew=True
    )

    # Execute Crew tasks
    result = my_crew.kickoff(inputs={'topic': topic_of_interest})
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    embedding = model.encode(result.raw)
    embedding = torch.tensor(embedding).float()

    return embedding

# Load job embeddings from the specified file path
def load_job_embeddings(file_path):
    return pd.read_pickle(file_path)


# Function to calculate job recommendations based on the resume embedding
def get_top_jobs(user_embedding, df_jobs, filters):
    similarities = []
    salary_differences = []

    # Iterate over each job embedding
    for index, row in df_jobs.iterrows():
        # Ensure that the embedding is correctly converted to a tensor
        job_emb = torch.tensor(row['embedding']).float()
        # Calculate cosine similarity
        sim = util.cos_sim(user_embedding, job_emb)
        similarities.append(sim.item())
        # Calculate the absolute difference from the user's desired salary
        salary_diff = abs(row['salary'] - filters['salary'])
        salary_differences.append(salary_diff)

    df_jobs['similarity'] = similarities
    df_jobs['salary_difference'] = salary_differences

    # Apply filters
    filtered_jobs = df_jobs.copy()

    # Apply hard filters
    if filters['job_titles']:
        job_titles_pattern = '|'.join([title.strip() for title in filters['job_titles']])
        filtered_jobs = filtered_jobs[filtered_jobs['job_title'].str.contains(job_titles_pattern, case=False, na=False)]

    if filters['company_names']:
        company_names_pattern = '|'.join([company.strip() for company in filters['company_names']])
        filtered_jobs = filtered_jobs[filtered_jobs['company'].str.contains(company_names_pattern, case=False, na=False)]

    if filters['locations']:
        locations_pattern = '|'.join([location.strip() for location in filters['locations']])
        filtered_jobs = filtered_jobs[filtered_jobs['job_location'].str.contains(locations_pattern, case=False, na=False)]

    if filters['salary'] > 0 :
        filtered_jobs = filtered_jobs[filtered_jobs['salary'] >= filters['salary']]

    # If no jobs match the filters, fall back to the top similar jobs
    if filtered_jobs.empty:
        # Sort by similarity and select top jobs, ignoring filters
        filtered_jobs = df_jobs.sort_values(by='similarity', ascending=False).head(10)

    # Sort the final results by similarity and salary difference
    final_jobs = filtered_jobs.sort_values(by=['similarity', 'salary_difference'], ascending=[False, True]).head(5)

    return final_jobs[['job_title', 'job_link', 'company', 'job_location', 'job_level', 'salary', 'similarity', 'job_summary', 'job_skills']]


def plot_top_job_titles(df_jobs):
    job_counts = df_jobs['job_title'].value_counts().head(10)
    fig = px.bar(job_counts, x=job_counts.index, y=job_counts.values,
                 labels={'x': 'Job Title', 'y': 'Count'},
                 title='Top 10 Job Titles',
                 color=job_counts.values,
                 color_continuous_scale='Blues',
                 width=700)  # Set chart width
    fig.update_layout(xaxis_tickangle=-45)  # Rotate X-axis labels
    st.sidebar.plotly_chart(fig, use_container_width=True)

def plot_top_locations(df_jobs):
    location_counts = df_jobs['job_location'].value_counts().head(10)
    fig = px.bar(location_counts, x=location_counts.index, y=location_counts.values,
                 labels={'x': 'Location', 'y': 'Count'},
                 title='Top 10 Job Locations',
                 color=location_counts.values,
                 color_continuous_scale='Viridis',
                 width=700)  # Set chart width
    fig.update_layout(xaxis_tickangle=-45)  # Rotate X-axis labels
    st.sidebar.plotly_chart(fig, use_container_width=True)

def plot_salary_distribution(df_jobs):
    fig = px.histogram(df_jobs, x='salary', nbins=30,
                       title='Salary Distribution',
                       labels={'salary': 'Salary'},
                       color_discrete_sequence=['#FFA07A'],
                       width=700)  # Set chart width
    st.sidebar.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Job Recommendation System")

    # User search query input with enhanced styling
    st.markdown("### What kind of job are you looking for today?")
    search_query = st.text_input(" ", placeholder="e.g., Data Analyst at VISA", label_visibility='collapsed')

    # File upload
    st.markdown("### Start with uploading your Resume!")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], label_visibility='collapsed')
    embedder = SentenceTransformer('all-mpnet-base-v2')


    # User input fields
    st.markdown("### What are your preferences?")
    salary_filter = st.slider("Minimum base Salary you're looking for(in $)", min_value=0, max_value=200000, value=50000, step=1000)
    location_filter = st.text_input("Desired Location (optional city name, comma-separated)", placeholder="e.g., Seattle, New York")
    job_title_filter = st.text_input("Desired Job Titles (optional, comma-separated)", placeholder="e.g., Data Analyst, Store Manager")
    company_name_filter = st.text_input("Desired Company Name (optional, comma-separated)", placeholder="e.g., VISA, Microsoft")


    # Load job embeddings from the specified file path
    job_embeddings_path = 'df_embed_test.pkl'  # Update to your path
    df_jobs = load_job_embeddings(job_embeddings_path)
    df_jobs['job_title'] = df_jobs['job_title'].apply(lambda x: x.title())

    # Sidebar to show the charts
    st.sidebar.title("Job Data Insights")
    plot_top_job_titles(df_jobs)
    plot_top_locations(df_jobs)
    plot_salary_distribution(df_jobs)

    if uploaded_file is not None:
        with st.spinner("Processing your inputs and your resume..."):
            # Process the uploaded resume and get the embedding and DataFrame
            embeddings = process_resume_with_agents(uploaded_file, search_query)

            # Get top job recommendations
            filters = {
                'salary': salary_filter,
                'locations': location_filter.split(','),
                'job_titles': job_title_filter.split(','),
                'company_names': company_name_filter.split(',')
            }

            if embeddings is not None:
                top_jobs = get_top_jobs(embeddings, df_jobs, filters)

            # Display the recommended jobs along with missing skills
            st.write("### Top 5 Job Recommendations:")
            if not top_jobs.empty:
              for index, job in top_jobs.iterrows():
                  st.write(f"#### {index+1}. [{job['job_title']}]({job['job_link']}) at {job['company']}")
                  st.write(f"**Location**: {job['job_location']}")
                  st.write(f"**Level**: {job['job_level']}")
                  st.write(f"**Salary**: ${job['salary']}")
                  st.write(f"**Similarity**: {job['similarity']:.2f}")
                  st.write(f"**Job Summary:** {job['job_summary'][:200]}...")  # Display first 200 characters of summary

                  # Find missing skills for each job
                  with st.spinner(f"Identifying missing skills for Job {index+1}..."):
                    missing_skills = find_missing_skills_for_job(uploaded_file, job['job_skills'])

                  if missing_skills:
                    st.write(f"**Missing Skills**: {', '.join(missing_skills)}")
                  else:
                    st.write("**Missing Skills**: None")

                  st.write("")
            else:
                st.write("No job recommendations found based on your criteria.")



if __name__ == "__main__":
    main()
