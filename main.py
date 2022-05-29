# Streamlit turns data scripts into shareable web apps in minutes
import streamlit as st
# to analyze data efficiently. Includes comparing, filtering, reshaping
import numpy as np
# to use open source data analysis and manipulation tool
import pandas as pd
# Seaborn helps to visualize the statistical relationships
import seaborn as sns
# import the image module from the pillow
from PIL import Image
# consist of several plots like line chart, bar chart, histogram, etc.
import matplotlib.pyplot as plt
# uses averaging to improve the predictive accuracy
from sklearn.ensemble import RandomForestRegressor
# to split data into training sets and test sets.
from sklearn.model_selection import train_test_split
# The warning filter in Python handles warnings
import warnings
warnings.filterwarnings("ignore")

# Data Loading
df_automobile = pd.read_csv("Automobile_data.csv")

#Data Cleaning(Data contains "?" replace it with NAN)
df_data = df_automobile.replace('?', np.NAN)
df_data.isnull().sum()

#fill missing data of normalised-losses, price, horsepower, peak-rpm, bore, stroke with the respective column mean
# Fill missing data category Number of doors with the mode of the column i.e. Four

df_temp = df_automobile[df_automobile['normalized-losses'] != '?']
normalised_mean = df_temp['normalized-losses'].astype(int).mean()
df_automobile['normalized-losses'] = df_automobile['normalized-losses'].replace('?', normalised_mean).astype(int)
df_temp = df_automobile[df_automobile['price'] != '?']
normalised_mean = df_temp['price'].astype(int).mean()
df_automobile['price'] = df_automobile['price'].replace('?', normalised_mean).astype(int)
df_temp = df_automobile[df_automobile['horsepower'] != '?']
normalised_mean = df_temp['horsepower'].astype(int).mean()
df_automobile['horsepower'] = df_automobile['horsepower'].replace('?', normalised_mean).astype(int)
df_temp = df_automobile[df_automobile['peak-rpm'] != '?']
normalised_mean = df_temp['peak-rpm'].astype(int).mean()
df_automobile['peak-rpm'] = df_automobile['peak-rpm'].replace('?', normalised_mean).astype(int)
df_temp = df_automobile[df_automobile['bore'] != '?']
normalised_mean = df_temp['bore'].astype(float).mean()
df_automobile['bore'] = df_automobile['bore'].replace('?', normalised_mean).astype(float)
df_temp = df_automobile[df_automobile['stroke'] != '?']
normalised_mean = df_temp['stroke'].astype(float).mean()
df_automobile['stroke'] = df_automobile['stroke'].replace('?', normalised_mean).astype(float)
df_automobile['num-of-doors'] = df_automobile['num-of-doors'].replace('?', 'four')

st.sidebar.title("Car Data Analysis")  #creating sidebar using streamlit
img = Image.open('car.jpg')
st.sidebar.image(img)                   #adding image to the sidebar

#Options to be selected
user_menu = st.sidebar.radio(
    'Lets Do Analysis on:',
    ("Cars Characteristics Frequency Analysis","Price Analysis","Correlation of various Characteristics","Highway Vs City mpg","Risky Vs Safe cars by body style","Analysis of Various Cars Features","Actual Value Vs Predicted Value For Car Price")
)


#Univariate Analysis
if user_menu == "Cars Characteristics Frequency Analysis":
    st.title("Analysis of Most cars in engine-size, peak-rpm ,curb-weight ,horsepower price")
    df = pd.DataFrame(df_automobile[:200], columns=['engine-size', 'peak-rpm', 'curb-weight', 'horsepower', 'price'])
    df.hist(figsize=(10, 8), bins=6, color='Blue')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    st.subheader("Most of the car has a Curb Weight is in range 1900 to 3100.")
    st.subheader("The Engine Size is inrange 60 to 190.")
    st.subheader("Most vehicle has horsepower 50 to 125.")
    st.subheader("Most Vehicle are in price range 5000 to 18000.")
    st.subheader("peak rpm is mostly distributed between 4600 to 5700")

    st.title("Number of Engine Type frequency diagram")
    df_automobile['engine-type'].value_counts(normalize=True).plot(figsize=(10, 8), kind='bar', color='black')
    plt.title("Number of Engine Type frequency diagram")
    plt.ylabel('Number of Engine Type')
    plt.xlabel('engine-type');
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    st.subheader("More than 70 % of the vehicle has Ohc type of Engine")

    st.title("Number of Door frequency diagram")
    df_automobile['num-of-doors'].value_counts(normalize=True).plot(figsize=(10, 8), kind='bar', color='blue')
    plt.title("Number of Door frequency diagram")
    plt.ylabel('Number of Doors')
    plt.xlabel('num-of-doors');
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    st.subheader("57% of the cars has 4 doors")

    st.title("Number of Fuel Type frequency diagram")
    df_automobile['fuel-type'].value_counts(normalize=True).plot(figsize=(10, 8), kind='bar', color='green')
    plt.title("Number of Fuel Type frequency diagram")
    plt.ylabel('Number of vehicles')
    plt.xlabel('fuel-type');
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    st.subheader("Gas is preferred by 85 % of the vehicles")

    st.title("Number of Body Style frequency diagram")
    df_automobile['body-style'].value_counts(normalize=True).plot(figsize=(10, 8), kind='bar', color='red')
    plt.title("Number of Body Style frequency diagram")
    plt.ylabel('Number of vehicles')
    plt.xlabel('body-style');
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    st.subheader("Most produced vehicle are of body style sedan around 48% followed by hatchback 32%")


#Bivariate Analysis
if user_menu == "Price Analysis":

    st.title("Price Analysis:")
    st.title("Car price by Various Company")
    st.subheader("Mercedez-Benz ,BMW, Jaguar, Porshe produces expensive cars more than 25000")
    st.subheader("cheverolet,dodge, honda,mitbushi, nissan,plymouth subaru,toyata produces budget models with lower prices")
    st.subheader("most of the cars comapany produces car in range below 25000")
    plt.rcParams['figure.figsize'] = (25, 15)
    sns.boxplot(x="make", y="price", data=df_automobile)
    plot_color = "#dd0033"
    st.pyplot()
    palette = ["#FFA07A", "#FF0000", "#dd0033", "#800000", "#DB7093"]
    fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0)

    st.subheader("Turbo models have higher prices than for the standard model, Convertible has only standard edition with expensive cars, hatchback and sedan turbo models are available below 20000")
    sns.catplot(data=df_automobile, x="make", y="price", hue="aspiration", kind="point")
    plt.xticks(rotation=90)
    st.pyplot()

    st.title("Mean Car Price by Various Company")
    ax.patch.set_alpha(0)
    grp1 = df_automobile.groupby('make')['price'].mean().reset_index()
    gax = sns.barplot(y='make', x='price', data=grp1, palette=palette, ax=ax)
    ax.set_xlabel('Mean Price', fontsize=14)
    ax.set_xlabel('', fontsize=14)
    st.pyplot(fig)

    st.title("Median car price by Various Company")
    plot_color = "#dd0033"
    palette = ["#FFA07A", "#FF0000", "#dd0033", "#800000", "#DB7093"]
    fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0.5)
    ax.set_title("Median Car Price by Company ", fontsize=16)
    ax.patch.set_alpha(0)
    grp1 = df_automobile.groupby('make')['price'].median().reset_index()
    gax = sns.barplot(y='make', x='price', data=grp1, palette=palette, ax=ax)
    ax.set_ylabel('Median Price', fontsize=14)
    ax.set_xlabel('', fontsize=14)
    st.pyplot(fig)

    st.title("Engine size Vs Price")
    fig, ax = plt.subplots(figsize=(6, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0.5)
    ax.set_title("Engine Size - Price", fontsize=16)
    ax.patch.set_alpha(0)
    gax5 = sns.regplot("engine-size", 'price', data=df_automobile, color=plot_color, ax=ax)
    gax5.set_ylabel('Price', fontsize=14)
    gax5.set_xlabel('Engine Size', fontsize=14)
    st.pyplot(fig)
    st.subheader("There is a positive correlation between Engine size and Price")

    st.title("Horsepower Vs Price")
    fig, ax = plt.subplots(figsize=(6, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0.5)
    ax.patch.set_alpha(0)
    gax8 = sns.regplot("horsepower", 'price', data=df_automobile, color='green')
    gax8.set_ylabel('Price', fontsize=14)
    gax8.set_xlabel('Horsepower', fontsize=14)
    st.pyplot(fig)
    st.subheader("There is a positive correlation between Horsepower and Price")

    st.title("Car Price by Body-style")
    st.subheader("Hardtop model are expensive in prices followed by convertible and sedan body style")
    plt.rcParams['figure.figsize'] = (20, 8)
    sns.boxplot(x="body-style", y="price", data=df_automobile)
    st.pyplot()


    st.title("Drive Wheels Vs Price")
    st.subheader("rwd wheel drive vehicle have expensive prices")
    plt.rcParams['figure.figsize'] = (12, 6)
    sns.boxplot(x="drive-wheels", y="price", data=df_automobile)
    st.pyplot()

    st.title("Engine Size Vs Horsepower Vs Price")
    gplt = sns.pairplot(df_automobile[['engine-size', 'horsepower', 'price']], kind='scatter', diag_kind='hist',diag_kws=dict(color='black', linewidth=1), plot_kws=dict(color='blue'))
    xlabels, ylabels = [], []
    for ax in gplt.axes[-1, :]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
    for ax in gplt.axes[:, 0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            gplt.axes[j, i].patch.set_alpha(0)
            gplt.axes[j, i].xaxis.set_label_text(xlabels[i], fontsize=14)
            gplt.axes[j, i].yaxis.set_label_text(ylabels[j], fontsize=14)
    st.pyplot()
    st.subheader("As the engine size and horsepower increases the price of the car also increases as there is positive correlation between them")

    st.title("Length Vs Width Vs Price")
    gplt = sns.pairplot(df_automobile[['length', 'width', 'price']], kind="scatter", diag_kind='scatter',diag_kws=dict(color=plot_color, linewidth=1), plot_kws=dict(color=plot_color))
    xlabels, ylabels = [], []
    for ax in gplt.axes[-1, :]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
    for ax in gplt.axes[:, 0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            gplt.axes[j, i].patch.set_alpha(0)
            gplt.axes[j, i].xaxis.set_label_text(xlabels[i], fontsize=14)
            gplt.axes[j, i].yaxis.set_label_text(ylabels[j], fontsize=14)
    st.pyplot()
    st.subheader("As the length and width of car increases price also increases because it increases the safe factor")

    st.title("Fuel Efficiency")
    fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0.5)
    ax.set_title("City MPG - Price", fontsize=16)
    ax.patch.set_alpha(0)
    gax9 = sns.regplot(x='city-mpg', y='price', data=df_automobile, color=plot_color)
    gax9.set_ylabel('Price', fontsize=14)
    gax9.set_xlabel('City MPG', fontsize=14)
    # gax9.set_xticklabels(gax9.get_xticklabels(), rotation=90)
    gax9.set_ylim(0, 50000)
    st.pyplot()
    st.subheader("Fuel Efficiency has a negatiVe correlation with price. People who look for highly fuel efficient car will normally be budget conscious. Hence it is probable that those cars are made for lower price brackets.")


if user_menu == "Correlation of various Characteristics":

    st.title("Correlation of various Characteristics")
    st.subheader("curb-size, engine-size, horsepower are positively correlated")
    st.subheader("city-mpg,highway-mpg are negatively correlated")
    corr = df_automobile.corr()
    plt.figure(figsize=(30, 20))
    sns.heatmap(corr, annot=True, fmt='.2f')
    st.pyplot()

    sns.heatmap(df_automobile[['wheel-base', 'symboling', 'height', 'num-of-doors']].corr(), cmap="BrBG", annot=True)
    st.pyplot()
    st.subheader("Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower (notice how all of these variables represent the size/weight/engine power of the car) Price is negatively correlated to ‘citympg’ and ‘highwaympg’ (-0.70 approximately).")
    st.subheader("Wheelbase and height are positively correlated")
    st.subheader("Wheelbase and Symboling are negatively correlated")
    st.subheader("height and symboling are negatively correlated")

if user_menu == "Highway Vs City mpg":
    fig, ax = plt.subplots(figsize=(4, 4), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0)
    ax.set_title("Highway Vs City MPG", fontsize=16)
    ax.patch.set_alpha(0)
    gax = sns.regplot("city-mpg", 'highway-mpg', data=df_automobile, color='blue')
    gax.set_ylabel('Highway-mpg', fontsize=14)
    gax.set_xlabel('City-mpg', fontsize=14)
    st.pyplot()
    st.subheader("city-mpg and highway-mpg are strongly correlated. For the purpose of comparison, we can use city-mpg as a realistic figure.")


if user_menu=="Risky Vs Safe cars by body style":
    palette = ["#FF0000", "#dd0033"]
    facet1 = sns.factorplot(x="symboling", y="normalized-losses", palette=palette, hue="body-style", kind="bar",data=df_automobile)
    facet1.facet_axis(0, 0).set_title("Risky Vs Safe Cars By Body Style", fontsize=16)
    facet1.facet_axis(0, 0).set_ylabel("Counts", fontsize=14)
    facet1.facet_axis(0, 0).set_xlabel("")
    facet1.facet_axis(0, 0).patch.set_alpha(0)
    facet1.facet_axis(0, 0).set_xticklabels(facet1.facet_axis(0, 0).get_xticklabels(), rotation=90, fontsize=14)
    facet1.fig.patch.set_alpha(0.5)
    st.pyplot()

    sns.catplot(data=df_automobile, y="normalized-losses", x="symboling", hue="body-style", kind="point")
    st.pyplot()
    st.subheader("Note :- here +3 means risky vehicle and -2 means safe vehicle")
    st.subheader("Increased in risk rating linearly increases in normalised losses in vehicle, covertible car and hardtop car has mostly losses with risk rating above 0, hatchback cars has highest losses at risk rating 3, sedan and Wagon car has losses even in less risk (safe)rating , There is a clear distinction in terms of insurance risk across body styles. Sedans and wagons are marked more safe compared to convertibles and hardtops. Hatchbacks on the other hand are a mix of everything.")

if user_menu =="Analysis of Various Cars Features":

    sns.factorplot(data=df_automobile, x="engine-type", y="engine-size", col="body-style", row="fuel-type")
    st.pyplot()
    st.subheader("ohc is the most used Engine Type both for diesel and gas")
    st.subheader("Diesel vehicle have Engine type ohc and I and engine size ranges between 100 to 190")
    st.subheader("Engine type ohcv has the bigger Engine size ranging from 155 to 300")
    st.subheader("Body-style Hatchback uses max variety of Engine Type followed by sedan")
    st.subheader("Body-style Convertible is not available with Diesel Engine type")

    sns.pairplot(df_automobile[["city-mpg", "horsepower", "engine-size", "curb-weight", "price", "fuel-type"]],hue="fuel-type", diag_kind="hist")
    st.pyplot()
    st.subheader("Vehicle Mileage decrease as increase in Horsepower , engine-size, Curb Weight")
    st.subheader("As horsepower increase the engine size increases")
    st.subheader("Curbweight increases with the increase in Engine Size")

    st.title("Number of cylinders Vs Horsepower")
    sns.catplot(data=df_automobile, x="num-of-cylinders", y="horsepower", kind="violin")
    st.pyplot()
    st.subheader("Vehicle with above 200 horsepower has Eight Twelve Six cyclinders")

    st.title("Wheel base on bodystyle")
    palette = ["#FF0000", "#bb1166"]
    fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)  # get the figure and axes objects
    fig.patch.set_alpha(0.5)
    ax.set_title("Wheelbase by Bodystyle", fontsize=16)
    ax.patch.set_alpha(0)
    gax = sns.swarmplot(x="body-style", y="wheel-base", hue="symboling", data=df_automobile, palette=palette)
    gax.set_ylabel('Wheel Base', fontsize=14)
    gax.set_xlabel('', fontsize=14)
    st.pyplot()
    st.subheader("Cars with long wheelbases tend to have better ride quality than those with short wheelbases.")


if user_menu == "Actual Value Vs Predicted Value For Car Price":

    #develop a model using predictor variables identified as important in exploratory data analysis.

    df_new = df_automobile[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','bore', 'wheel-base', 'city-mpg', 'length', 'width', ]]

    ##Splitting data into training and testing with ratio of 80:20 respectively

    x_train, x_test, y_train, y_test = train_test_split(df_new, df_automobile['price'], test_size=0.20, random_state=1)
    print("number of test samples :", x_test.shape[0])
    print("number of training samples:", x_train.shape[0])

    # Random Forest is non linear regression based on decision trees.

    Rf = RandomForestRegressor()
    Rf.fit(x_train, y_train)
    predicted = Rf.predict(x_test)
    plt.figure(figsize=(15, 5))
    ax1 = sns.distplot(df_automobile['price'], hist=False, color="r", label="Actual Value")
    sns.distplot(predicted, hist=False, color="b", label="predicted Values", ax=ax1)
    st.title('Actual vs predicted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    st.pyplot()
    st.subheader("The R_squared value for Random Forest Regression Model is:  0.8795538677279676.")
    st.subheader("We can say that ~ 87% of the variation of the price is explained by this random forest model")
