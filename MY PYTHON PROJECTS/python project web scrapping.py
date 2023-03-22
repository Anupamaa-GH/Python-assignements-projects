#!/usr/bin/env python
# coding: utf-8

# In[4]:


#imoprt the library to query a website
import requests
#specify the URL
wiki_link="https://en.wikipedia.org/wiki/List_of_countries_in_the_Americas_by_population"
link=requests.get(wiki_link).text


# In[5]:


print(link)


# In[7]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(link, 'lxml')
print(soup)


# In[8]:


print(soup.prettify())


# In[9]:


soup.title


# In[10]:


# to have title without strinng
soup.title.string


# In[11]:


#check the links
soup.a


# In[12]:


#to find all links
soup.find_all("a")


# In[14]:


#to get all href  Link
all_link=soup.find_all("a")
for link in all_link:
    print(link.get("href"))


# In[15]:


# extract of all tables
all_tables=soup.find_all('table')
print(all_tables)


# In[16]:


#to extract  the right table
right_table=soup.find('table', class_='sortable wikitable')
right_table


# In[18]:


#to find correct links from table (a tag)
table_links = right_table.findAll('a')
table_links


# In[19]:


# to  get all list of countries
country=[]
for links in table_links:
    country.append(links.get('title'))
print(country)


# In[22]:


#represent into data frame
import pandas as pd
df =pd.DataFrame()
df['Country']=country
df


# In[ ]:





# In[ ]:




