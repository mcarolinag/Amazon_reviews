#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:34:44 2018

@author: carolina

info: this code extracts review info and scrapes Amazon.com for item prices.
"""


from lxml import html  
from lxml import html  
import csv,os,json
import requests
from time import sleep
import random
import json 

def extract():
    import csv
    import numpy as np
    import pandas as pd


    observations= pd.read_csv('amazon_reviews_us_Sports_v1_00.tsv', sep='\t', header=0, error_bad_lines=False)
    
    observations.drop_duplicates(inplace=True)
    
    observations.dropna(inplace=True)
    
    return observations



def AmzonParser(url):
    from lxml import html    
    import csv,os,json
    import requests
    import random
    import json
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    page = requests.get(url,headers=headers)
    while True:
        sleep(random.randrange(1,3))
        try:
            doc = html.fromstring(page.content)
            XPATH_NAME = '//h1[@id="title"]//text()'
            XPATH_SALE_PRICE = '//span[contains(@id,"ourprice") or contains(@id,"saleprice")]/text()'
            XPATH_ORIGINAL_PRICE = '//td[contains(text(),"List Price") or contains(text(),"M.R.P") or contains(text(),"Price")]/following-sibling::td/text()'
            XPATH_CATEGORY = '//a[@class="a-link-normal a-color-tertiary"]//text()'
            XPATH_AVAILABILITY = '//div[@id="availability"]//text()'

            RAW_NAME = doc.xpath(XPATH_NAME)
            RAW_SALE_PRICE = doc.xpath(XPATH_SALE_PRICE)
            RAW_CATEGORY = doc.xpath(XPATH_CATEGORY)
            RAW_ORIGINAL_PRICE = doc.xpath(XPATH_ORIGINAL_PRICE)
            RAw_AVAILABILITY = doc.xpath(XPATH_AVAILABILITY)
            
            NAME = ' '.join(''.join(RAW_NAME).split()) if RAW_NAME else None
            SALE_PRICE = ' '.join(''.join(RAW_SALE_PRICE).split()).strip() if RAW_SALE_PRICE else None
            CATEGORY = ' > '.join([i.strip() for i in RAW_CATEGORY]) if RAW_CATEGORY else None
            ORIGINAL_PRICE = ''.join(RAW_ORIGINAL_PRICE).strip() if RAW_ORIGINAL_PRICE else None
            AVAILABILITY = ''.join(RAw_AVAILABILITY).strip() if RAw_AVAILABILITY else None

            if not ORIGINAL_PRICE:
                ORIGINAL_PRICE = SALE_PRICE

            if page.status_code!=200:
                raise ValueError('captha')

            data = {
					'NAME':NAME,
					'SALE_PRICE':SALE_PRICE,
					'CATEGORY':CATEGORY,
					'ORIGINAL_PRICE':ORIGINAL_PRICE,
					'AVAILABILITY':AVAILABILITY,
					'URL':url,
					}

            return data
        except Exception as e:
            print(e)
            break
            
def ReadAsin(AsinList):
 
    from time import sleep
    import csv,os,json

    for i in range(len(AsinList)):
        url = "http://www.amazon.com/dp/"+AsinList[i]
        print("Processing: "+url)
        extracted_data.append(AmzonParser(url))
        sleep(random.randrange(1, 5))
        f=open('data.json','w')
        json.dump(extracted_data,f,indent=4)
    return extracted_data

AsinList=list(words_topic2['product_id'].values)


f=open('data.json','w')
json.dump(data,f,indent=4)



    
if __name__ == "__main__":
    read_reviews():
    avoid_captha(frequent_pidlist)
