#!/usr/bin/env python
# coding: utf-8

# In[18]:


# 17-9-23
# CSC461 – Assignment1 – Web Scraping 
# Maryam Yousaf
# FA21-BSE-077
# Question2 Part 1 Scrapped birthdate of those similiar to your birthdate from the site provided and export it to text file
# Question2 Part 2 Scrapped important events that occured on your birthdate from the site provided and export it to text file
from bs4 import BeautifulSoup
import requests


# In[19]:


link = "https://www.timeanddate.com/on-this-day"
my_Birthday = "october/16"
complete_link = f"{link}/{my_Birthday}"
print(complete_link)


# In[20]:


response = requests.get(complete_link)
soup = BeautifulSoup(response.text, 'html.parser')


# In[21]:


if(response.status_code == 200):
    main_Div = soup.find('article', class_ = "otd-row otd-life")
    outer_Div = main_Div.find('h3', class_ = "otd-life__title")
    #print("Actual text of outer_div:", outer_Div.text.strip())
    if outer_Div and outer_Div.text.strip()  == "Births On This Day, October 16":
        inner_Div = main_Div.find('ul',class_ = "list--big")
        double_Div = inner_Div('li')
        with open('Part1.txt', 'w') as file:
            file.write("On 16 October The following people share BirthDate \n")
            for BirthDates in double_Div:
                    print(BirthDates.text)
                    written = BirthDates.text + '\n' 
                    file.write(written)
    else:
        print("Nothing Found")


# In[22]:


link2 = "https://www.britannica.com/on-this-day"
my_Birthday2 = "october-16"
complete_link2 = f"{link2}/{my_Birthday2}"
print(complete_link2)


# In[23]:


response2 = requests.get(complete_link2)
soup2 = BeautifulSoup(response2.text, 'html.parser')


# In[24]:


if(response2.status_code == 200):
    outer_Div2 = soup2.find('div', class_= 'card-body')
    head_Div2 = outer_Div2.find('div', class_= 'title font-18 font-weight-bold mb-10').text
    text_Div = outer_Div2.find('div', class_= 'description font-serif')
    with open('Part1.txt', 'a',encoding='utf-8') as file:
            file.write("\nOn my birthDate The following events happened:\n")
            written = '\n' + head_Div2 + '\n' + text_Div.text 
            print(written)
            file.write(written)
            file.close()
    second_Div = soup2.find_all('div', class_ = 'card-body font-serif')
    with open('Part1.txt', 'a',encoding='utf-8') as file:
               file.write("\nMore Events That Ocuured on my birthDate:\n")
               for Events in second_Div:
                            indented_Data = "\n" + Events.text.replace('\n', '\n ')
                            print(indented_Data)
                            file.write(indented_Data)
else:
    print("Incorrect-url")


# In[ ]:




