# app.py 
import os
import asyncio
import re
import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Scraper function
async def scrape_properties():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://propertyonion.com/property_search")
        await page.wait_for_timeout(10000)

        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        property_cards = soup.find_all('div', class_='p-card-content')

        data = []
        for card in property_cards:
            try:
                status = card.find('div', class_='p-chip')
                status = status.text.strip() if status else 'N/A'

                date_divs = card.select('div.flex.flex-wrap.text-right > div.px-2.pt-1')
                date = date_divs[-1].text.strip() if date_divs else 'N/A'

                address_tag = card.find('div', class_='ellip')
                address_lines = address_tag.find_all('span') if address_tag else []
                address = ' '.join([line.text.strip() for line in address_lines]) if address_lines else 'N/A'

                full_info = address_tag.get_text(separator=" ", strip=True) if address_tag else ''
                beds_match = re.search(r'(\d+)\s+Beds', full_info)
                baths_match = re.search(r'(\d+)\s+Baths', full_info)
                sqft_match = re.search(r'([\d,]+)\s+sqft', full_info)

                beds = beds_match.group(1) if beds_match else 'N/A'
                baths = baths_match.group(1) if baths_match else 'N/A'
                sqft = sqft_match.group(1).replace(',', '') if sqft_match else 'N/A'

                deal_tag = card.find('span', class_='ng-star-inserted')
                deal_type = deal_tag.text.strip() if deal_tag else 'N/A'

                if deal_type.lower() == 'for':
                    deal_type = 'N/A'

                data.append({
                    'Status': status,
                    'Date': date,
                    'Deal Type': deal_type,
                    'Address': address,
                    'Beds': beds,
                    'Baths': baths,
                    'Sqft': sqft
                })
            except Exception as e:
                print("Error parsing a card:", e)

        df = pd.DataFrame(data)
        df.to_csv('listings.csv', index=False)
        return df

# LangChain LLM setup
def setup_agent(df):
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4")
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )

# Prompt templates
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

# Streamlit UI
st.title("🏡 Property Listings AI Assistant")

# Scrape data button
if st.button("🔄 Scrape Fresh Data"):
    with st.spinner("Scraping data, please wait..."):
        df = asyncio.run(scrape_properties())
        st.success("✅ Scraping complete!")
else:
    try:
        df = pd.read_csv("listings.csv").fillna(value=0)
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please scrape data first.")
        st.stop()

# Display preview
st.write("### Preview of Listings")
# st.dataframe(df.head())

# Add this
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Listings as CSV",
    data=csv,
    file_name='property_listings.csv',
    mime='text/csv'
)

# Ask a question
st.write("### Ask the Agent a Question")
question = st.text_input("What would you like to know about the listings?",
                         "Which city or area has the highest number of property listings?")

if st.button("Run Query"):
    agent = setup_agent(df)
    query = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    with st.spinner("Running AI Agent..."):
        res = agent.invoke(query)
        st.markdown("### Final Answer")
        st.markdown(res["output"]) 




