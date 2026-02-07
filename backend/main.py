import pandas as pd
import numpy as np
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def Crime_input_processing(ip):
    # text, image, video, audio, form

    data = {
        'Year': 0, 'Amount_Lost_INR': 0, 'City': '', "Category": '',
        'Ransomware': 0, 'Data Breach': 0, 'Hacking': 0, 'Malware': 0,
        'Identity Theft': 0, 'Phishing': 0, 'Online Fraud': 0,
        'Cyber Bullying': 0, 'Others': 0
    }

    incidents = {
        'Ransomware': 0, 'Data Breach': 0, 'Hacking': 0, 'Malware': 0,
        'Identity Theft': 0, 'Phishing': 0, 'Online Fraud': 0,
        'Cyber Bullying': 0, 'Others': 0
    }
    print(data.keys())
    
    if ip.get('text'):
        print("text")
        template = '''Task: You are a data extraction specialist. Analyze the provided complaint text and extract the following fields into a structured string format each field seprated by $.
Fields to Extract: Year$Amount_Lost_INR$City$Category$Type
Year: The 4-digit year when the incident occurred. If not found, use "NULL".
Amount_Lost_INR: The total monetary loss in Indian Rupees. Extract only the number. If the amount is in another currency, convert it to INR if possible; otherwise, provide the number and note the currency.
City: The city where the incident took place.
Category: Map the incident to exactly one of the following categories based on the target or nature of the crime:
Social Media, Government, Corporate, E-commerce, Education, Financial, Personal, Health.
Type: Map the crime to exactly one of the following categories of the crime:
Ransomware, Data Breach, Hacking, Malware, Identity Theft, Phishing, Online Fraud, Cyber Bullying, or Others.
Constraint: If a field is not mentioned in the text, return NULL for that field.
Complaint Text: {text}.'''
        
        query = PromptTemplate.from_template(template)
        chain = query | llm | StrOutputParser()
        text_op = chain.invoke({'text': ip['text']})
        print(text_op)
        print(ip['text'])

    if ip.get('audio'):
        template = '''Task: You are a data extraction specialist. Analyze the provided complaint audio and extract the following fields into a structured string format each field seprated by $.
Fields to Extract:
Year: The 4-digit year when the incident occurred. If not found, use "NULL".
Amount_Lost_INR: The total monetary loss in Indian Rupees. Extract only the number. If the amount is in another currency, convert it to INR if possible; otherwise, provide the number and note the currency.
City: The city where the incident took place.
Category: Map the incident to exactly one of the following categories based on the target or nature of the crime:
Social Media, Government, Corporate, E-commerce, Education, Financial, Personal, Health.
Type: Map the crime to exactly one of the following categories of the crime:
Ransomware, Data Breach, Hacking, Malware, Identity Theft, Phishing, Online Fraud, Cyber Bullying, or Others.
Constraint: If a field is not mentioned in the text, return NULL for that field.
Complaint Text: {audio}.'''

        with open(ip['audio'], "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": template},
                {"type": "media", "mime_type": "audio/mpeg", "data": audio_base64}
            ]
        )
        audio_op = llm.invoke([message])
        print(f"Audio Extraction: {audio_op.content}")

    if ip.get('image'):
        template = '''Task: You are a data extraction specialist. Analyze the provided complaint in image and extract the following fields into a structured string format each field seprated by $.
Fields to Extract:
Year: The 4-digit year when the incident occurred. If not found, use "NULL".
Amount_Lost_INR: The total monetary loss in Indian Rupees. Extract only the number. If the amount is in another currency, convert it to INR if possible; otherwise, provide the number and note the currency.
City: The city where the incident took place.
Category: Map the incident to exactly one of the following categories based on the target or nature of the crime:
Social Media, Government, Corporate, E-commerce, Education, Financial, Personal, Health.
Type: Map the crime to exactly one of the following categories of the crime:
Ransomware, Data Breach, Hacking, Malware, Identity Theft, Phishing, Online Fraud, Cyber Bullying, or Others.
Constraint: If a field is not mentioned in the text, return NULL for that field.
Complaint Text: {image}.'''
        
        image_data = encode_image(ip['image'])
        message = HumanMessage(
            content=[
                {"type": "text", "text": template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        # Note: ChatPromptTemplate is harder to use with binary data; invoke LLM directly for simplicity
        image_op = llm.invoke([message])
        print(f"Image Extraction: {image_op.content}")

    if ip['form']:
        pass




if __name__ == "__main__":

    data = {}
    data['text'] = None
    data['audio'] = None
    data['video'] = None
    data['form'] = None
    data['image'] = '/Users/shivamg/Downloads/WhatsApp Image 2026-02-06 at 12.10.42.jpeg'
    Crime_input_processing(data)

    # ip_type = int(input('Type: '))
    # if ip_type:
    #     print("llm is responding")
    #     text = '''Today is 2026, On 2023 I Received sms and then I had call the number they told me to download from play store kyc anydesk app I had downloaded then they told to do next and number arrived on the screen they had asked that no I told them digit no but then I realised that it is a fraud call I had cut the call and informed Vodafone customer care and also informed on police number they told me file a complaint'''

    #     template = '''Task: You are a data extraction specialist. Analyze the provided complaint text and extract the following fields into a structured string format each field seprated by $.
    #         Fields to Extract:
    #         Year: The 4-digit year when the incident occurred. (If not found, use "Unknown").
    #         Amount_Lost_INR: The total monetary loss in Indian Rupees. Extract only the number. If the amount is in another currency, convert it to INR if possible; otherwise, provide the number and note the currency.
    #         City: The city where the incident took place.
    #         Category: Map the incident to exactly one of the following categories based on the target or nature of the crime:
    #             Social Media, Government, Corporate, E-commerce, Education, Financial, Personal, Health.
    #         Type: Map the crime to exactly one of the following categories of the crime:
    #             Ransomware, Data Breach, Hacking, Malware, Identity Theft, Phishing, Online Fraud, Cyber Bullying, or Others.
    #         Constraint: If a field is not mentioned in the text, return null for that field.
    #         Complaint Text: {text}'''
    #     query = PromptTemplate.from_template(template)
    #     chain = query | llm | StrOutputParser
    #     output = chain.invoke({'text': text})
    #     print(output)

    # else:
    #     # year = int(input("year: "))
    #     # amount_lost = int(input("Amount_Lost_INR: "))
    #     # city = input("City: ")
    #     # category = input("Organisation: ")
    #     # incident_type = input("Incident_Type: ")

    #     # data = {
    #     #     'Year': 0, 'Amount_Lost_INR': 0, 'City': '', "Category": '',
    #     #     'Ransomware': 0, 'Data Breach': 0, 'Hacking': 0, 'Malware': 0,
    #     #     'Identity Theft': 0, 'Phishing': 0, 'Online Fraud': 0,
    #     #     'Cyber Bullying': 0, 'Others': 0
    #     # }

    #     # incidents = {
    #     #     'Ransomware': 0, 'Data Breach': 0, 'Hacking': 0, 'Malware': 0,
    #     #     'Identity Theft': 0, 'Phishing': 0, 'Online Fraud': 0,
    #     #     'Cyber Bullying': 0, 'Others': 0
    #     # }
    #     # print(data.keys())

    #     # if incident_type in incidents.keys():
    #     #     data[incident_type]=1 
    #     # else:
    #     #     data['Others'] = 1

    #     # path = "/Users/shubham/Desktop/Projects/CyberCrime Hackathon/data/cyber_crime_1.csv"
    #     # df = pd.read_csv(path)

    #     if df['Year'][0] == year:
    #         for incident in incidents.keys():
    #             # print(incident, end = ' ')
    #             if data[incident]==1:
    #                 data[f"prev_{incident}"] = df.loc[0,f"prev_{incident}"] + 1
    #             else:
    #                 data[f"prev_{incident}"] = df.loc[0,f"prev_{incident}"]

    #     else:
    #         for incident in incidents.keys():
    #             data[f"prev_{incident}"] = data[incident]


    # print(data)
