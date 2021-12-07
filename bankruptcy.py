
# Import pandas and load in the data
import pandas as pd
import numpy as np
df = pd.read_csv("combined.csv")


# Let's explore the data
# Tells you the number of rows by columns
print(df.shape)


# Let's take a look at our three columns of interest
print(df.loc[:, ["State of Debtor", "Population by sex and state and year", "Average Weekly Pay by Sex and State and year"]])


# Dropping some row values that don't apply to Australia 
df.drop(df.loc[df["State of Debtor"]=="Unknown"].index, inplace=True)
df.drop(df.loc[df["State of Debtor"]=="International"].index, inplace=True)


# Let's find the greatest average population and see which state it's in
pop_by_state = df.groupby("State of Debtor")["Population by sex and state and year"].mean().reset_index()
pop_by_state.round(decimals = 2)
# Clearly, New South Wales has the greatest average population


# Here's the overall average population
round(df["Population by sex and state and year"].mean(), 2)


# Which state has the most bankruptcy records
print(df["State of Debtor"].value_counts())
# New South Wales with 116,705 records


# Let’s see if any state is disproportionately affected by bankruptcy 
# In other words, what percentage of their population has bankruptcy records
average_pop = df.groupby("State of Debtor")["Population by sex and state and year"].mean().astype(int).to_dict()
data_points = df["State of Debtor"].value_counts().to_dict()
ratio = {}
for key in data_points:
    for key in average_pop:
        if key == key:
            ratio[key] = round((data_points[key]/average_pop[key]), 5)
            
for key in sorted(ratio):
    print(key, ":", ratio[key])
# All are relatively similar (ranges from 2%-4%), suggests no disproportionate effect


# Let's move onto analysing the weekly pay column
# Checking the datatype
print(df.dtypes)


# The column is an object so we must convert to float so we can aggregate
df["Average Weekly Pay by Sex and State and year"] = df["Average Weekly Pay by Sex and State and year"].str.replace(",", "").astype(float)


# Now let's find the state with the highest average, average weekly pay 
pays_by_state = df.groupby("State of Debtor")["Average Weekly Pay by Sex and State and year"].mean().reset_index()
pays_by_state.round(decimals = 2)


# What's the average, average weekly pay
round(df["Average Weekly Pay by Sex and State and year"].mean(), 2)


# Let’s find the most frequently occuring state for each income bracket
df.groupby("State of Debtor")["Average Weekly Pay by Sex and State and year"].value_counts(bins = [500, 700, 900, 1100, 1300, 1500])


# Let’s find the number of data points for each income bracket
df["Average Weekly Pay by Sex and State and year"].value_counts(bins = [500, 700, 900, 1100, 1300, 1500]).sort_values(ascending=True)


# Let’s move onto the third chart 
# Creating a dictionary for what percentage of the total population each state is 
av_pop_by_state = df.groupby("State of Debtor")["Population by sex and state and year"].mean().to_dict()
sum_of_av_pops = df.groupby("State of Debtor")["Population by sex and state and year"].mean().sum()
percent_of_total_pop = {}
for key in av_pop_by_state:
    percent_of_total_pop[key] = round(av_pop_by_state[key]/sum_of_av_pops, 3)


# Now for income, what percentage of the total weekly income is each state
av_income_by_state = df.groupby("State of Debtor")["Average Weekly Pay by Sex and State and year"].mean().to_dict()
sum_of_av_incomes = df.groupby("State of Debtor")["Average Weekly Pay by Sex and State and year"].mean().sum()
percent_of_total_income = {}
for key in av_income_by_state:
   percent_of_total_income[key] = round(av_income_by_state[key]/sum_of_av_incomes, 3)


# Now for bankruptcy records, what percentage of the total records is each state 
total_records = 356500
records_by_state = df["State of Debtor"].value_counts().to_dict()
percent_of_total_records = {}
for key in records_by_state:
    percent_of_total_records[key] = round(records_by_state[key]/total_records, 3)
# The percentages of these dictionaries add up to 1, 0.99, and 0.995


# Import some modules and packages 
import texthero as hero
from texthero import preprocessing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
import plotly.express as px
from texthero import stopwords


# Let's check what the column looks like
# Preparing for our word cloud
df["Main Cause of Insolvency"].head()


# Clean the column
cleaning = [preprocessing.fillna, preprocessing.lowercase, preprocessing.remove_digits, preprocessing.remove_punctuation]
df["Main Cause of Insolvency"] = hero.clean(df["Main Cause of Insolvency"], pipeline = cleaning)
df["Main Cause of Insolvency"].head()


# Let's remove words that don't add any insight like or, of, on
default_stopwords = stopwords.DEFAULT
extra_stopwords = default_stopwords.union(set(["of","or", "and", "on"]))
df["Main Cause of Insolvency"] = hero.remove_stopwords(df["Main Cause of Insolvency"], extra_stopwords)


# Let's create a graph showing the frequency of words in this column
# Shows the most common causes of bankruptcy 
# This visualisation is interactive 
data = hero.visualization.top_words(df["Main Cause of Insolvency"]).head(20)
fig = px.bar(data)
fig.update_layout(title="Causes of Bankruptcy", xaxis_title="Bankruptcy-related Words", yaxis_title="Number of Occurrences")
fig.show()
data.head()


# Let's make a word cloud
# Will display the above results in a more visually appealing way
wordcloud = WordCloud(width = 800, height = 800, background_color = "black",  max_words = 1000, min_font_size = 20).generate(str(df["Main Cause of Insolvency"]))
fig = plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Let’s create the bar plot
# Set width, height, and position on x-axis
barWidth = 0.25
Population = [0.017, 0.321, 0.01, 0.201, 0.072, 0.022, 0.249, 0.107]
Income = [0.151, 0.12, 0.144, 0.12, 0.11, 0.104, 0.115, 0.136]
Bankruptcies = [0.011, 0.327, 0.007, 0.272, 0.062, 0.028, 0.205, 0.083]
br1 = np.arange(len(Population))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


# Create the plot
# This chart compares four attributes (three numeric, one categorical)
plt.bar(br1, Population, color="darkblue", width=barWidth, edgecolor="none", label="Population")
plt.bar(br2, Income, color="gainsboro", width=barWidth, edgecolor="none", label="Income")
plt.bar(br3, Bankruptcies, color="lightsalmon", width=barWidth, edgecolor="none", label="Bankruptcies")
plt.xlabel("States", fontweight="light", fontsize=15)
plt.ylabel("Percent of Total", fontweight="light", fontsize=15)
plt.xticks([r + barWidth for r in range(len(Population))], ["ACT", "NSW", "NT", "QLD", "SA", "Tas", "Vic", "WA"]) 
plt.legend()
plt.show()

