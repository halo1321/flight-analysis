from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from google.cloud import bigquery
from google.cloud import storage
import os
import json
import tempfile
import pandas as pd
from datetime import date
import tempfile
import streamlit as st
from PyPDF2 import PdfReader


def format_to_html(data):
    if isinstance(data, dict):
        html = "<ul>"
        for key, value in data.items():
            html += f"<li><strong>{key}:</strong> {format_to_html(value)}</li>"
        html += "</ul>"
    elif isinstance(data, list):
        html = "<ul>" + "".join(f"<li>{format_to_html(item)}</li>" for item in data) + "</ul>"
    else:
        html = f"<p>{data}</p>"
    return html

 
if st.button('Clear', type="primary"):
    st.session_state.messages.clear()



st.title("🤖 Hi, My Name is Rudy, Your Assistant for Route Analysis")
st.subheader("Conversation")



main_prompt = """
Persona:
You are "Rudy", an expert in Route Network Development for Airlines and Airports, specializing in aviation industry strategies with over 10 years of professional experience. Your expertise lies in crafting comprehensive strategies for optimizing air service, enhancing connectivity, and leveraging data-driven insights.
Your role is to act as both a strategic advisor and mentor, providing in-depth detailed, actionable recommendations and justifications for your recommendations to improve route network development and air service strategies.
Your responses Your responses should simulate how a senior professional would guide a new team member, explaining not only the "what" but also the "why" and "how" behind each decision. Incorporate a teaching approach that includes practical examples, detailed analysis, and industry-specific insights with incorporate industry knowledge, market trends, and best practices in route network planning, ensuring alignment with IATA standards and aviation regulations.
You will provide the summary, reccommedation, and reason for the new route creation based on the information provided.
 
You will:
Generate comprehensive route development strategies tailored to specific airline and airport needs.
Provide actionable insights on market demand, aircraft utilization, route profitability, and competition analysis.
Recommend optimal airport pairs, potential operator airlines from both Origin Country and Destination Country, aircraft types, and scheduling strategies based on current and future market trends.
Ensure that your suggestions are realistic, actionable, and designed to maximize both airline and airport profitability while addressing passenger needs.
 
Detailed Explanations: Break down each element of your recommendation, explaining the rationale and interconnections, as if educating someone new to the field.
In-Depth Analysis: Include comprehensive evaluations of market dynamics, operational feasibility, and strategic alignment, with supporting data and assumptions clearly articulated.
Real-World Context: Reference practical aviation scenarios, trends, and case studies to enhance understanding.
Key Insights & Risks: Highlight critical insights, potential risks, and their mitigation, explaining their significance in the decision-making process.
Step-by-Step Guidance: Offer clear steps for implementation, explaining how to monitor and adapt strategies over time.
 
Your responses should be structured, clear, and presented in a professional tone. Key aspects to include in recommendations:
Summary: Provide an overview of the recommendation in clear and concise terms, setting the stage for the detailed explanation.
Recommendation: Present the proposed strategy in actionable terms, with explicit suggestions on routes, airlines, aircraft, and other specifics.
Reason: Deliver an in-depth explanation of the recommendation. Address each of the following aspects:
Market Analysis: Include passenger demand numbers (if available), travel behavior insights, and competition details. If data is missing, explain how assumptions were derived.
Operational Feasibility: Discuss the suitability of aircraft, scheduling, and infrastructure needs.
Strategic Alignment: Explain how the recommendation aligns with broader goals (e.g., airline objectives, government policies, or regional development plans).
Risks and Mitigation: Elaborate on potential risks, providing clear mitigation strategies.
Use case Examples: Use real-world aviation scenarios, such as similar routes or historical trends, to reinforce understanding.
 
Format the result as a JSON object and using "Summary", "Recommendation", "Reason" as key.
 
###
Description
 
"News"
Definition: A continuously updated source of information obtained from CAPA (web service) or similar platforms. It includes the latest updates on route openings, frequency adjustments, and airline announcements.
Purpose: Provides real-time insights into industry developments, enabling stakeholders to monitor which routes are operational or have undergone recent changes.
Role: Helps route developers stay informed about market movements and align strategies with current trends or opportunities.
 
"Analysis Report"
Definition: A document containing in-depth information such as governance policies, market trends, regional growth forecasts, and airline performance data.
Purpose: Provides comprehensive contextual knowledge that complements real-time news updates. It serves as a foundation for strategic planning and decision-making.
Role: "Enables a holistic view of potential route opportunities by integrating with News, considering both qualitative and quantitative factors.
 
"Demands"
Definition: Represents the passenger or cargo demand between city pairs, derived from market analysis or demand forecasting tools.
Purpose: Identifies potential route opportunities and estimates their profitability. However, demand data alone does not confirm the existence of a route.
Role: Cross-referenced with Airlines Schedule Change to verify whether a route currently exists or represents a new development opportunity.
 
"Airlines Schedule Change"
Definition: A list detailing the changes in airline schedules (city pairs) over a defined period, such as between 2022 and 2024.
Purpose: Highlights new routes, discontinued services, or frequency adjustments, enabling route developers to focus on gaps and emerging opportunities.
Role: Instrumental in identifying new routes for development by confirming that the proposed routes do not already exist in the latest schedules.
 
"Date"
Definition: The current date used in conjunction with RAG (Retrieval-Augmented Generation) systems to analyze data dynamically.
Purpose: Ensures the framework remains up-to-date by filtering News and Airlines Schedule Change to identify potential routes that are not currently operational.
Role: Supports real-time validation of non-existent routes, allowing for accurate and timely recommendations.
 
####
Example
 
News: US FAA may upgrade Thailand's aviation safety rating to Category 1 in Feb-2025: Minister
Analysis Report: Ignite Thailand, the governance policy to boosting tourism of Thailand
Demand: LAX  has a demands to all Thailand airport around 180,000 person a year
Airline Schedule Change: There no route between Los Angeles and Bangkok, This will be a opportunities
Date: 4 Nov 2024
 
User Query: Can you suggest potential new route between Thailand and USA
Summary:
Recommendation: LAX-BKK by Thai Airways using Boeing 787-900 to operates this routes
Reason: Because there are not have direct routes between Los Angeles and Bangkok. And there are high demand between it and the news tell us there will be eligible to operate direct flight after Feb-2025
####
 
News: {news_text}
Analysis Report: {analysis_report_text}
Demand: {damand}
Airline Schedule Change: {schedule}
Date: {today_date}
 
User Query: {user_query}
Summary:
Recommendation:
Reason:
"""
 
demand_table_schema = f"""
table_metadata =
    table_name: Demand-2024,
    description: This table contains demand-related information for airline routes, including origin and destination details, passenger counts, revenue, fare, and market share data for 2024.,
    columns:
        ID:
            data_type: integer,
            description: Unique identifier for each record.,
            example_value: 1
        ,
        Orig:
            data_type: string,
            description: Airport code for the origin location.,
            example_value: ICN
        ,
        Orig_Desc:
            data_type: string,
            description: Full description of the origin airport.,
            example_value: Incheon International
        ,
        Dest:
            data_type: string,
            description: Airport code for the destination location.,
            example_value: BKK
        ,
        Dest_Desc:
            data_type: string,
            description: Destination airport or city name.,
            example_value: Suvarnabhumi Int'l
        ,
        Orig_Country:
            data_type: string,
            description: Country code for the origin location.,
            example_value: KR, US, UK
        ,
        Orig_Country_Desc:
            data_type: string,
            description: Full name of the origin country.,
            example_value: South Korea, USA, UK
        ,
        Dest_Country:
            data_type: string,
            description: Country code for the destination location.,
            example_value: TH
        ,
        Dest_Country_Desc:
            data_type: string,
            description: Full name of the destination country.,
            example_value: Thailand
        ,
        Total_Pax:
            data_type: integer,
            description: Total number of passengers between the origin and destination locations for 11 months.,
            example_value: 2,186,398
        ,
        Pax_Share:
            data_type: float,
            description: Percentage of passenger share for the route.,
            example_value: 2.75%
        ,
        Fare:
            data_type: float,
            description: Average fare for the route.,
            example_value: 211.89
        ,
        Rev:
            data_type: integer,
            description: Total revenue generated for the route.,
            example_value: 463,285,709
        ,
        Total_Pax_12month:
            data_type: integer,
            description: Total passenger in year that convert from original value 11 months to 12 months.,
            example_value: 2,385,161
"""
 
demand_big_query_prompt = """
You are a sophisticated BigQuery SQL query generator.
Translate the following natural language request (human query) into a valid BigQuery syntax (SQL query).
Consider the table schema provided.
FROM always `madt-is-445507.madt_dataset.demand-2024`
Format the SQL Query result as JSON with 'big_query' as a key.
 
###
Example:
table_name: Demand-2024,
description: This table contains demand-related information for airline routes, including origin and destination details, passenger counts, revenue, fare, and market share data for 2024.,
columns:
    ID:
        data_type: integer,
        description: Unique identifier for each record.,
        example_value: 1
    ,
    Orig:
        data_type: string,
        description: Airport code for the origin airport or origin location.,
        example_value: ICN
    ,
    Orig_Desc:
        data_type: string,
        description: Full description of the origin airport or origin location.,
        example_value: Incheon International
 
Human Query: Ranking the highest total passenger to least
 
SQL Query: SELECT Orig, Dest, SUM(CAST(REPLACE(Total_Pax_12month, ',', '') AS INT)) AS Total_Pax_12month
FROM `madt-is-445507.madt_dataset.demand-2024`
GROUP BY Orig, Dest
ORDER BY Total_Pax_12month DESC;
 
###
Table Schema: {demand_table_schema}
Human Query: {query}
SQL Query:
"""
 
schedule_extract_country_prompt = """
Extract country from the query exclude Thailand
Format the result as JSON with 'Country as a key.
###
Query: {query}
Country:
"""
 
schedule_table_schema = f"""
table_metadata =
    table_metadata =
        table_name: Airline_Schedule_Change,
        description: This table contains data on airline schedule changes, including airline details, aircraft deployed, route information, and frequency changes between October 2022 and March 2025.,
        columns:
            Airline_Name:
                data_type: string,
                description: Name of the airline operating the route.,
                example_value: Aeroflot
            ,
            Aircraft_Deployed:
                data_type: string,
                description: Type of aircraft deployed for the route.,
                example_value: Boeing 737-800 (winglets) Passenger
            ,
            Departure_Airport:
                data_type: string,
                description: Full name of the departure airport.,
                example_value: Irkutsk International Airport
            ,
            Departure_Airport_Code:
                data_type: string,
                description: IATA code for the departure airport.,
                example_value: IKT
            ,
            Departure_Airport_Country:
                data_type: string,
                description: Full name of the origin country.,
                example_value: Russian Federation
            ,
            Arrival_Airport:
                data_type: string,
                description: Full name of the arrival airport.,
                example_value: Bangkok Suvarnabhumi International Airport
            ,
            Arrival_Airport_Code:
                data_type: string,
                description: IATA code for the arrival airport.,
                example_value: BKK
            ,
            Arrival_Airport_Country:
                data_type: string,
                description: Full name of the destination country.,
                example_value: Thailand
            ,
            Frequency_of_Week_Starting_31-Oct-2022:
                data_type: integer,
                description: Weekly frequency of flights as of October 31, 2022.,
                example_value: 7
            ,
            Frequency_of_Week_Starting_31-Mar-2025:
                data_type: integer,
                description: Weekly frequency of flights as of March 31, 2025.,
                example_value: 0
            ,
            Frequency_Change:
                data_type: integer,
                description: Change in weekly flight frequency between the two periods.,
                example_value: -7
            ,
            Routes:
                data_type: string,
                description: Route code representing the concatenate of IATA code between departure airports and arrival airports.,
                example_value: IKTBKK
"""
 
schedule_big_query_prompt = """
You are a sophisticated BigQuery SQL query generator.
Translate the following natural language request (human query) into a valid BigQuery syntax (SQL query).
Consider the table schema provided.
FROM always `madt-is-445507.madt_dataset.airline_schedule_change.
Format the SQL Query result as JSON with 'big_query' as a key.
 
###
Example:
table_name: Airline_Schedule_Change,
description: This table contains data on airline schedule changes, including airline details, aircraft deployed, route information, and frequency changes between October 2022 and March 2025.,
columns:
    Airline_Name:
        data_type: string,
        description: Name of the airline operating the route.,
        example_value: Aeroflot
    ,
    Aircraft_Deployed:
        data_type: string,
        description: Type of aircraft deployed for the route.,
        example_value: Boeing 737-800 (winglets) Passenger
    ,
    Departure_Airport:
        data_type: string,
        description: Full name of the departing airport.,
        example_value: Irkutsk International Airport
 
Human Query: Ranking the popular job from most to least popular
 
SQL Query: SELECT Arrival_Airport, COUNT(*) AS ArrivalCount
FROM `madt-is-445507.madt_dataset.airline_schedule_change`
GROUP BY Arrival_Airport
ORDER BY ArrivalCount DESC;
 
###
Table Schema: {schedule_table_schema}
Human Query: Distinct the route from Thailand to {country}, also show Departure Airport and Arrival Airport, and order by asc
SQL Query:
"""
 
analysis_prompt = """
Persona:
You are "Rudy", an expert in Route Network Development for Airlines and Airports, specializing in aviation industry strategies with over 10 years of professional experience.
Your expertise lies in crafting comprehensive strategies for optimizing air service, enhancing connectivity, and leveraging data-driven insights.
Using the user's query and metadata, you will generate new insightful and unique analysis questions for 5 new related questions.
The objective of the new 5 questions is to find the new route.
The total of passengers from the user query is value information, so you must have 3 out of 5 questions.
Format the reponse as JSON with "analysis_question" as a key and "question" as a sub key:
####
 
Schema:
{demand_table_schema}
####
 
Example: Can you suggest potential new route between Spain and Thailand
User query:
Question:
'analysis_question': 'question': ['Considering the existing flight routes from Spain to other Asian destinations and the passenger demand from Spain to Thailand, which Spanish airports show the highest potential for a new route to Thailand, based on underserved demand and existing connectivity?',
   'Analyzing the passenger volume and revenue data from existing routes between Thailand and other European countries, what is the estimated potential demand for a new route between Spain and Thailand, and which specific cities in Spain and Thailand would be most viable?',
   'Based on the average fare and revenue data for existing routes between Spain and other long-haul destinations, and between Thailand and other European destinations, what is a realistic pricing strategy for a new Spain-Thailand route to ensure profitability, considering seasonal variations and competition?',
   'Considering the current capacity and operational constraints of Spanish and Thai airports, which airport pairs (in Spain and Thailand) would be most suitable for a new route, considering factors such as runway length, terminal capacity, and ground handling capabilities?',
   'By analyzing the seasonal trends in passenger demand for existing routes to and from Spain and Thailand, what is the optimal time of year to launch a new route to maximize passenger numbers and revenue, and what marketing strategies could be employed to attract passengers during off-peak seasons?']
####
 
User query: {user_query}
Question:
"""
 
today_date = str(date.today())
project_id = 'madt-is-445507'
parser = JsonOutputParser()
 
def gemini_model(google_api_key):
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-002', temperature=0, google_api_key=google_api_key)
    return model
 
def rag(openai_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store
 
def read_news(df_news, openai_api_key):
    # df_news = pd.read_csv("/content/news.csv")
    loader = DataFrameLoader(df_news, page_content_column="Contents")
    docs = loader.load()
    vector_store = rag(openai_api_key=openai_api_key)
    return vector_store
 
def news(vector_store, user_query):
    news = vector_store.similarity_search(query=user_query, k=3)
    news_text = ''
    for i in range(len(news)):
        news_text += str('News_' + str(i+1) + ': ' + news[0].metadata['Date'] + ' ' + news[0].metadata['Headline'] + ' ' + news[i].page_content)
    return news_text
 
def report(pdf):
    # loader = PyPDFLoader("/content/Outlines vision to Ignite Tourism Thailand.pdf")
 
    class PDFLoader:
        def __init__(self, pdf_reader):
            self.pages = pdf_reader.pages
        def load(self):
            return [{"page_content": page.extract_text()} for page in self.pages]
       
    loader = PDFLoader(pdf)
    docs = loader.load()
    # analysis_report_text = ''
    # for i in range(len(docs)):
    #     analysis_report_text += docs[i].page_content
    analysis_report_text = ''
    for i in range(len(docs)):
        analysis_report_text += docs[i]['page_content']
    return analysis_report_text
 
def gen_question(model, user_query):
    parser = JsonOutputParser()
    analysis_prompt_template = PromptTemplate(template=analysis_prompt, input_variables=['user_query', 'demand_table_schema'])
    analysis_prompt_chain = analysis_prompt_template | model | parser
    analysis_prompt_result = analysis_prompt_chain.invoke({"user_query": user_query, "demand_table_schema": demand_table_schema})
    return analysis_prompt_result
 
def schedule(model, client, query):
    #Country Extraction
    extract_shcedule_bigquery_prompt_template = PromptTemplate(template=schedule_extract_country_prompt, input_variables=['query'])
    extract_shcedule_bigquery_chain = extract_shcedule_bigquery_prompt_template | model | parser
    extract_shcedule_sql_bigquery_result = extract_shcedule_bigquery_chain.invoke({"query": query})
    country = extract_shcedule_sql_bigquery_result['Country']
    print(country)
 
    #Schedule
    shcedule_bigquery_prompt_template = PromptTemplate(template=schedule_big_query_prompt, input_variables=['schedule_table_schema', 'country'])
    shcedule_bigquery_chain = shcedule_bigquery_prompt_template | model | parser
    shcedule_sql_bigquery_result = shcedule_bigquery_chain.invoke({"schedule_table_schema": schedule_table_schema, "country": country})
    print('---' * 20)
    print(shcedule_sql_bigquery_result)
    shcedule_bigquery_query = shcedule_sql_bigquery_result['big_query']
    shcedule_bigquery_query_result = client.query(shcedule_bigquery_query).to_dataframe()
    return shcedule_bigquery_query_result
 
def demand(analysis_prompt_result, model, client, query):
    #Demand
    print(analysis_prompt_result)
    demand_bigquery_prompt_template = PromptTemplate(template=demand_big_query_prompt, input_variables=['demand_table_schema', 'query'])
    demand_bigquery_chain = demand_bigquery_prompt_template | model | parser
    demand_sql_bigquery_result = demand_bigquery_chain.invoke({"demand_table_schema": demand_table_schema, "query": query})
    demand_bigquery_query = demand_sql_bigquery_result['big_query']
    demand_bigquery_query_result = client.query(demand_bigquery_query).to_dataframe()
    # demand_df_list = []
    # for i in range(len(analysis_prompt_result['analysis_question']['question'])):
    #   if i == 0:
    #     demand_sql_bigquery_result = demand_bigquery_chain.invoke({"demand_table_schema": demand_table_schema, "query": query})
    #     demand_bigquery_query = demand_sql_bigquery_result['big_query']
    #     demand_bigquery_query_result = client.query(demand_bigquery_query).to_dataframe()
    #     demand_df_list.append(demand_bigquery_query_result)
    #   else:
    #     demand_sql_bigquery_result = demand_bigquery_chain.invoke({"demand_table_schema": demand_table_schema, "query": analysis_prompt_result['analysis_question']['question'][i]})
    #     print(demand_sql_bigquery_result)
    #     demand_bigquery_query = demand_sql_bigquery_result['big_query']
    #     demand_bigquery_query_result = client.query(demand_bigquery_query).to_dataframe()
    #     demand_df_list.append(demand_bigquery_query_result)
    return demand_bigquery_query_result
 
def main_run(model, user_query, news_text, analysis_report_text, demand_df_list, shcedule_bigquery_query_result):
    main_prompt_template = PromptTemplate(template=main_prompt, input_variables=['news_text',
                                                                              'analysis_report_text',
                                                                              'damand',
                                                                              'schedule',
                                                                              'today_date'])
    main_prompt_chain = main_prompt_template | model | parser
    main_prompt_result = main_prompt_chain.invoke({'user_query': user_query,
                                                    'news_text': news_text,
                                                    'analysis_report_text': analysis_report_text,
                                                    'damand': demand_df_list,
                                                    'schedule': shcedule_bigquery_query_result,
                                                    'today_date': today_date})
    return main_prompt_result
 
def main():
 
    with st.sidebar:
        st.title(":red[Credential and Key]")
 
        csv_uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if not csv_uploaded_file:
            st.info("Please upload news to continue.", icon="🗝️")
        if csv_uploaded_file is not None:
            # if st.button("Add News"):
            df_news = pd.read_csv(csv_uploaded_file)
              #st.write("Here are the contents of your CSV file:")
 
        pdf_uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if not pdf_uploaded_file:
            st.info("Please upload report to continue.", icon="🗝️")
        if pdf_uploaded_file is not None:
            # if st.button("Add Report"):
            pdf_reader = PdfReader(pdf_uploaded_file)
              #st.write("Here are the contents of your CSV file:")
 
        uploaded_file = st.file_uploader("Upload Credential File .json", type="json")
        if not uploaded_file:
            st.info("Please add your Bigquery creditial to continue.", icon="🗝️")
        if uploaded_file is not None:
            if st.button("Add Creditial"):
              with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                  temp_file.write(uploaded_file.read())
                  temp_file_path = temp_file.name
 
              os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
              st.success("Bigquery creditial successfully uploaded.", icon="✅")
 
        google_api_key = st.text_input("Gemini API Key", type="password")
        if not google_api_key:
            st.info("Please add your Gemini API key and Bigquery creditial to continue.", icon="🗝️")
        if google_api_key is not None:
            if st.button("Add Gemini API Key"):
                st.success("Gemini API key successfully uploaded.", icon="✅")
 
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        else:
            if st.button("Add OpenAI API Key"):
                st.success("OpenAI API key successfully uploaded.", icon="✅")
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    if user_input := st.chat_input("What can I help you with today? 😊"):
 
        model = gemini_model(google_api_key=google_api_key)
        vector_store = rag(openai_api_key=openai_api_key)
        client = bigquery.Client(project=project_id)
 
 
        vector_store = read_news(df_news=df_news, openai_api_key=openai_api_key)
        news_text = news(vector_store=vector_store,
                         user_query=user_input)
        analysis_report_text = report(pdf=pdf_reader)
        analysis_prompt_result = gen_question(model=model,
                                              user_query=user_input)
        demand_df_list = demand(analysis_prompt_result=analysis_prompt_result,
                                model=model,
                                client=client,
                                query=user_input)
        shcedule_bigquery_query_result = schedule(model=model,
                                                  client=client,
                                                  query=user_input)
        main_prompt_result = main_run(model=model,
                                        user_query=user_input,
                                        news_text=news_text,
                                        analysis_report_text=analysis_report_text,
                                        demand_df_list=demand_df_list,
                                        shcedule_bigquery_query_result=shcedule_bigquery_query_result)
       
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": main_prompt_result})
 
 
        chat_css = """
            <style>
            .chat-container {
                display: flex;
                align-items: flex-start;
                margin: 10px 0;
            }
            .user-message {
                margin-right: auto;
                background-color: #fce4ec;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .assistant-message {
                margin-left: auto;
                background-color: #fff9c4;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                background-color: #f5f5f5;
                border-radius: 50%;
                font-size: 20px;
                margin: 0 10px;
            }
            .user-container {
                display: flex;
                flex-direction: row-reverse;
                align-items: center;
            }
            .assistant-container {
                display: flex;
                flex-direction: row;
                align-items: center;
            }
            </style>
            """
        st.markdown(chat_css, unsafe_allow_html=True)
        for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-container user-container">
                            <div class="user-message">{format_to_html(message['content'])}</div>
                            <div class="icon">👤</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-container assistant-container">
                            <div class="icon">🤖</div>
                            <div class="assistant-message">{format_to_html(message['content'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


 
if __name__ == '__main__':
    main()