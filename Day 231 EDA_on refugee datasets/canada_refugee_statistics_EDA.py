# Canada Refugee Statistics Exploratory Data Analysis

"""
### Questions
- From which countries has Canada admitted the highest number of refugees?
- What are the total number of resettled refugees in Canada per year?
- What are the countries of origin for the majority of asylum claims made in Canada?
- What is the total number of asylum claims made in Canada every year?
- What are the general trends in refugee and asylum statistics from 2012-2022? 
"""

# - Step 1: Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# - Step 2: Reading, Exploring and Preparing data

data = pd.read_csv('can-stats-2012-22.csv')

data.head()

data.info()

# checking missing values
pd.DataFrame(data.isnull().sum(),columns=['Missing values'])

# - Step 3: Exploratory Data Analysis + Visualization

# Question 1: Countries From Which Canada Has Admitted Highest Number of Refugees

# Create a DataFrame with the top countries and their UNHCR-refugees sum, then sort values in desc order
top_unhcr_refugees = data.groupby('Country-of-origin')['UNHCR-refugees'].sum().reset_index(name='UNHCR-refugees').sort_values(by='UNHCR-refugees', ascending=False).head(10)

# Display the DataFrame with the top countries for UNHCR Refugees
top_unhcr_refugees.reset_index(drop=True).style.bar()

# Create Seaborn bar graph to visualize top 10 countries with highest number of refugees resettled in Canada
plt.figure(figsize=(10,6))
sns.barplot(data=top_unhcr_refugees, x='Country-of-origin', y='UNHCR-refugees', palette='mako')

# Add labels and titles
plt.xlabel('Country of Origin')
plt.ylabel('Resettd refugees')
plt.title('Top Countries of origin for Resettled Refugees in Canada (2012-2022)')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Display plot
plt.tight_layout()
plt.show()

data.groupby('Country-of-origin')['UNHCR-refugees'].sum().sort_values(ascending=False).head(10)

toprefugees = data.groupby('Country-of-origin')['UNHCR-refugees'].sum().sort_values(ascending=False).head(10).reset_index(name='Total')

plt.xticks(rotation=45)
sns.barplot(toprefugees,x='Country-of-origin',y='Total')

# Question 2: Total Number of Resettled Refugees in Canada Per Year

total_refugees_yrly = data.groupby('Year')['UNHCR-refugees'].sum().reset_index(name='UNHCR-refugees').sort_values(by='Year')

total_refugees_yrly.reset_index(drop=True)

# Create a Seaborn line graph to visualize total number of resettled refugees in Canada per year
plt.figure(figsize=(10, 6))
sns.lineplot(data=total_refugees_yrly, x='Year', y='UNHCR-refugees', marker='o')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Resettled Refugees')
plt.title('Total Number of Resettled Refugees (2012-2022)')

# Display plot
plt.tight_layout()
plt.show()

# Question 3: Countries of Origin for Majority of Asylum Claims Made in Canada

# Create a DataFrame with the top countries and their Asylum-seekers sum, then sort values in desc order
top_asylum_seekers = data.groupby('Country-of-origin')['Asylum-seekers'].sum().reset_index(name='Asylum-seekers').sort_values(by='Asylum-seekers', ascending=False).head(10)

# Display the DataFrame with the top countries for Asylum-seekers
top_asylum_seekers.reset_index(drop=True).style.bar()

# Create Seaborn bar graph to visualize top 10 countries of origin for asylum seekers in Canada
plt.figure(figsize=(10,6))
sns.barplot(data=top_asylum_seekers, x='Country-of-origin', y='Asylum-seekers', palette='mako')

# Add labels and title
plt.xlabel('Country of Origin')
plt.ylabel('Asylum Seekers')
plt.title('Countries of Origin for Majority of Asylum Seekers from 2012-2022')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Display plot
plt.tight_layout()
plt.show()

# Question 4: Total Number of Asylum Claims Made in Canada Every Year

asylum_seekers_yrly = data.groupby('Year')['Asylum-seekers'].sum().reset_index(name='Asylum-seekers').sort_values(by='Year')

asylum_seekers_yrly.reset_index(drop=True)

# Create a Seaborn line graph visualize total number of asylum being made in Canada every year
plt.figure(figsize=(10, 6))
sns.lineplot(data=asylum_seekers_yrly, x='Year', y='Asylum-seekers',marker='o')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Count of Asylum Seekers')
plt.title('Total Asylum Claims from 2012-2022')

# Display the plot
plt.tight_layout()
plt.show()

# Question 5: General Trends in Refugee and Asylum Statistics from 2012-2022

# Countries with highest overall count
top_countries = data.groupby('Country-of-origin')['total-count'].sum().reset_index(name='total-count').sort_values(by ='total-count', ascending=False).head(10)

top_countries.reset_index(drop=True).style.bar()

# Create bar graph to visualize which countries have the overall highest count from 2012-2022 of both refugees and asylum seekers
plt.figure(figsize=(10,6))
sns.barplot(data=top_countries, x='Country-of-origin', y='total-count', palette='mako')

# Create labels and title
plt.xlabel('Country of Origin')
plt.ylabel('Total')
plt.title('Countries With Majority of Refugees and Asylum Seekers from 2012-2022')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Display plot
plt.tight_layout()
plt.show()

# General trends in number of refugees and asylum seekers

# Group the data by year and calculate the sum of each category for each year
yearly_counts = data.groupby('Year')[['UNHCR-refugees', 'Asylum-seekers']].sum().reset_index()

# Pivot the data to create a suitable format for bar plotting
melted_data = pd.melt(yearly_counts, id_vars=['Year'], var_name='Category', value_name='Count')

# Create a bar graph with unstacked, side-by-side bars using Matplotlib
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_data, x='Year', y='Count', hue='Category', palette='mako')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Total of Refugees and Asylum Seekers from 2012-2022')

# Display the legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# - 2012- 2022 Canada Refugee Statistics EDA Results Summary
"""
- Countries From Which Canada Has Admitted Highest Number of Refugees:

Colombia (108,416)

China (98,586)

Ukraine (88,376)

Pakistan (74,737)

Haiti (70,956)

Sri Lanka (66,343)

Nigeria (60,554)

Mexico (51,072)

Türkiye (42,533)

Iran (40,881)
"""

"""
- Total Number of Resettled Refugees in Canada Per Year:

2012: 163,751

2013: 160,347

2014: 149,164

2015: 135,890

2016: 97,322

2017: 104,768

2018: 114,101

2019: 101,757

2020: 109,214

2021: 130,125
"""

"""
- Countries of Origin for Majority of Asylum Claims Made in Canada:

Unknown (56,098)

Nigeria (51,620)

India (48,806)

Mexico (47,146)

Haiti (40,908)

Colombia (30,475)

China (21526)

Pakistan (19,023)

Türkiye (18,166)

Iran (15,011)
"""

"""
- Total Number of Asylum Claims Made in Canada Every Year:

2012: 32,647

2013: 22,145

2014: 16,699

2015: 19,631

2016: 23,946

2017: 51,859

2018: 78,766

2019: 97,017

2020: 85,352

2021: 63,143

2022: 113,066
"""

"""
- General Trends in Refugee and Asylum Statistics from 2012-2022:

2012 was the year where Canada admitted the highest number of refugees, followed by 2013, 2014 and 2022.

2016 had the lowest number of resettled refugees

2022 had the highest number of asylum claims made in Canada, followed by 2019 and 2020

2014 had the lowest number of asylum claims, followed by 2015 and 2013
"""