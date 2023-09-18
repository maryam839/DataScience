#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 17-9-23
# CSC461 – Assignment1 – Web Scraping 
# Maryam Yousaf
# FA21-BSE-077
# Scrapped title and rating of 5 favourite movie of yours from imdb site and export it to excel file


# In[2]:


import requests
from bs4 import BeautifulSoup
import time
import openpyxl
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}


# In[3]:


movie_1 = requests.get("https://www.imdb.com/title/tt1375666/?ref_=nv_sr_srsg_0_tt_8_nm_0_q_incep", headers=headers)


# In[4]:


movie_2 = requests.get("https://www.imdb.com/title/tt2737304/?ref_=nv_sr_srsg_1_tt_7_nm_0_q_birdbo",headers=headers)


# In[5]:


movie_3 = requests.get("https://www.imdb.com/title/tt10872600/?ref_=nv_sr_srsg_0_tt_8_nm_0_q_spiderman%2520no%2520way%2520home",headers=headers)


# In[6]:


movie_4 = requests.get("https://www.imdb.com/title/tt5814060/?ref_=nv_sr_srsg_4_tt_7_nm_0_q_the%2520nu",headers=headers)


# In[7]:


movie_5 = requests.get("https://www.imdb.com/title/tt1187043/?ref_=fn_al_tt_1",headers=headers)


# In[8]:


fav_movies = [movie_1,movie_2,movie_3,movie_4,movie_5]


# In[9]:


Excel_file = openpyxl.Workbook()
Excel_sheet = Excel_file.active
Excel_sheet.title = "5 Movies Rating"
print(Excel_file.sheetnames)
Excel_sheet.append(['Movie Name','IMDB Rating'])


# In[10]:


for movies in fav_movies:
    html_content = movies.text
    soup = BeautifulSoup(html_content, 'html.parser')
    Outer_div = soup.find('div', class_ = 'sc-e226b0e3-3 jJsEuz')
    if Outer_div:
        Inner_div = Outer_div.find('div', class_ = 'sc-dffc6c81-0 iwmAVw')
        Title = Inner_div.find('span', class_ = 'sc-afe43def-1 fDTGTb')
        Rating = Outer_div.find('span', class_ = 'sc-bde20123-1 iZlgcd')
        if Title: 
            title = Title.text
            print(title)
        if Rating:
            rating =float(Rating.text) 
            print(rating)
        else:
            print("No element found")
    time.sleep(1)
    Excel_sheet.append([title,rating])


# In[11]:


Excel_file.save("5 Movies Rating.xlsx")


# In[ ]:




