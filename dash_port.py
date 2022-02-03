import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import re
import os
import datetime
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
geolocator = Nominatim(user_agent='myuseragent')
import plotly.express as px
from PIL import Image
import pydeck as pdk
import random

############################
#Setup some random Geo data around area
############################

Ottawa = [45.3833982402789, -75.67276727227754]
Barrhaven = [45.272872635070634, -75.73915984822567]
Kanata = [45.31222088889224, -75.91535486940379]

lat_O= [random.uniform(Ottawa[0]-0.035, Ottawa[0]+0.03) for i in range(1350)]
lon_O = [random.uniform(Ottawa[1]-0.04, Ottawa[1]+0.03) for i in range(1350)]
lat_B= [random.uniform(Barrhaven[0]-0.045, Barrhaven[0]+0.045) for i in range(1600)]
lon_B = [random.uniform(Barrhaven[1]-0.045, Barrhaven[1]+0.045) for i in range(1600)]
lat_K = [random.uniform(Kanata[0]-0.045, Kanata[0]+0.045) for i in range(2050)]
lon_K = [random.uniform(Kanata[1]-0.05, Kanata[1]+0.045) for i in range(2050)]


lat2_O= [random.uniform(Ottawa[0]-0.035, Ottawa[0]+0.03) for i in range(400)]
lon2_O = [random.uniform(Ottawa[1]-0.04, Ottawa[1]+0.03) for i in range(400)]
lat2_B= [random.uniform(Barrhaven[0]-0.045, Barrhaven[0]+0.045) for i in range(550)]
lon2_B = [random.uniform(Barrhaven[1]-0.045, Barrhaven[1]+0.045) for i in range(550)]
lat2_K = [random.uniform(Kanata[0]-0.045, Kanata[0]+0.045) for i in range(850)]
lon2_K = [random.uniform(Kanata[1]-0.05, Kanata[1]+0.045) for i in range(850)]

dataO2 = pd.DataFrame({'Location':'Ottawa',
                      'lat':lat2_O,
                     'lon': lon2_O})
dataB2 = pd.DataFrame({'Location':'Barrhaven',
                      'lat':lat2_B,
                     'lon': lon2_B,
                     })
dataK2 = pd.DataFrame({'Location':'Kanata',
                      'lat':lat2_K,
                     'lon': lon2_K})

df_loc_2 = pd.concat([dataO2, dataB2, dataK2]).reset_index()


dataO = pd.DataFrame({'Location':'Ottawa',
                      'lat':lat_O,
                     'lon': lon_O})
dataB = pd.DataFrame({'Location':'Barrhaven',
                      'lat':lat_B,
                     'lon': lon_B,
                     })
dataK = pd.DataFrame({'Location':'Kanata',
                      'lat':lat_K,
                     'lon': lon_K})

df_loc = pd.concat([dataO, dataB, dataK]).reset_index()

#####################
#Create business data
#####################


services = ['Service #1', 'Service #2', 'Service #3', 'Sevice #4']
customers = ['New', 'Recurring', 'Non-Recurring']
price_dict = dict(zip(services, [125, 155, 170, 265]))
c_data = random.choices(customers, weights=[1, 10, 5], k = 5000)
s_data = random.choices(services, weights=[10, 7, 5, 1 ], k = 5000)

df_dash = pd.DataFrame({'Customer':c_data,
                        'Service': s_data,
                        })
df_dash['Price'] = df_dash.Service.map(price_dict)

########################
## Add some dates
########################

data = df_dash.join(df_loc)
month = random.choices(range(5, 8), weights=(22,41,37), k=5000)
day = random.choices(range(1, 31), k=5000)
year = [2021 for i in range(5000)]

df = pd.DataFrame({'year': year,
                   'month': month,
                   'day': day})
data['Date'] = pd.to_datetime(df[['year', 'month', 'day']])


#######################
## Function for Mapping
#######################

def map(data, lat, lon, zoom):
    st.pydeck_chart(pdk.Deck(
        map_style="road",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 45,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=150,
                elevation_scale=4,
                elevation_range=[-10, 800],
                pickable=True,
                extruded=True,
            ),
        ]
    ))



######################
## Streamlit Design // Panel
#####################

st.set_page_config(
    page_title="JFM-Data Dash",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)


st.sidebar.header('Select Parameters')


S_Location = st.sidebar.selectbox('Select Location',['All']+list(pd.unique(data.Location)))


S_Service = st.sidebar.selectbox('Select Service',['All']+list(pd.unique(data.Service)))

######################
## Streamlit Design // Main Page
#####################

st.title('Example Subscription Dashboard')



######################
## Streamlit Design // Main Page
#####################

dict_loc = {"All" : [45.356, -75.6, 10.5],
            "Ottawa" : [45.3833982402789, -75.67276727227754, 11],
            "Barrhaven" : [45.272872635070634, -75.73915984822567, 11],
            "Kanata" : [45.31222088889224, -75.91535486940379, 11]}
    

st.dataframe(data)


row_1, row_2, row_3,  = st.columns((1,1,1))

with row_1:
    st.write("**Ottawa South Location**")
    map(df_loc_2, Ottawa[0], Ottawa[1], 11)

with row_2:
    st.write("**Barrhaven Location**" )
    map(df_loc_2, Barrhaven[0], Barrhaven[1], 11)

with row_3:
    st.write("**Kanata Location**" )
    map(df_loc_2, Kanata[0], Kanata[1], 11)



######################
## Display Map
#####################
 
st.header('Trailing 3 Months Revenue')

row2_1, row2_2  = st.columns((5,1))

 



#fig_Units_team = px.area(data, x="Date", y="Price")
#                           facet_row="Total", template='plotly_dark')
st.plotly_chart(fig_Units_team, use_container_width=True)
chart1 = data.loc[:,['Price', 'Date']].groupby('Date').sum().cumsum()

with row2_1:
    st.area_chart(chart1, use_container_width=True,)


with row2_2:
    st.title("")
    st.title("Total Revenue: $**%i**" % (sum(data.Price)))

st.text("Customer Visits By Week")   
st.bar_chart(data=data.groupby(data.Date.dt.strftime('%W')).Customer.count())

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])

chart_data = pd.pivot_table(data, values='Price', index='Customer', aggfunc=np.sum)
chart_data = data.groupby([data.Date.dt.strftime('%W'), 'Customer']).Customer.count()

test1 = data.groupby([data.Date.dt.strftime('%W'), 'Customer']).Customer.count()
chart4 = test1.unstack(level=1)

import altair as alt
st.text("Revenue by Customer Type")   
chart4    =    alt.Chart(data).mark_bar().encode(

                y='sum(Price)',
                x='Date',
                color='Customer',
            )

st.altair_chart(chart4, use_container_width=True)




data.groupby(df.Date.dt.strftime('%W')).Customer.count()


with row3_1:
     st.title("Yesterday")
     st.header("🚀 New: \t \t \t     **%i** " % (data.loc[:,['New', 'Date']].groupby('Date').sum()[-1:].sum()))
     st.header(":small_red_triangle_down:  Renewals:\t\t **%i** " % (data.loc[:,['Renewals', 'Date']].groupby('Date').sum()[-1:].sum()))
     st.header("🚀 Cancelled:\t\t **%i**" % (data.loc[:,['Cancelled', 'Date']].groupby('Date').sum()[-1:].sum()))
    
with row3_2:
     st.title("Last 7 Days")
     st.header("🚀 New: \t \t \t     **%i** " % (data.loc[:,['New', 'Date']].groupby('Date').sum()[-7:].sum()))
     st.header("🚀 Renewals:\t\t **%i**" % (data.loc[:,['Renewals', 'Date']].groupby('Date').sum()[-7:].sum()))
     st.header("🔻 Cancelled:\t\t **%i** " % (data.loc[:,['Cancelled', 'Date']].groupby('Date').sum()[-7:].sum()))
    
with row3_3:
     st.title("Last 30 Days")
     st.header(":small_red_triangle_down: New: \t \t \t     **%i** " % (data.loc[:,['New', 'Date']].groupby('Date').sum()[-30:].sum()))
     st.header("🚀 Renewals:\t\t **%i** " % (data.loc[:,['Renewals', 'Date']].groupby('Date').sum()[-30:].sum()))
     st.header("🚀 Cancelled:\t\t **%i**" % (data.loc[:,['Cancelled', 'Date']].groupby('Date').sum()[-30:].sum()))

map(data_map, 'Ottawa', 11)
import time


# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))

# for i in range(100):
#     # Update progress bar.
#     progress_bar.progress(i + 1)

#     new_rows = np.random.randn(10, 2)

#     # Update status text.
#     status_text.text(
#         'The latest random number is: %s' % new_rows[-1, 1])

#     # Append data to the chart.
#     chart.add_rows(new_rows)

#     # Pretend we're doing some computation that takes time.
#     time.sleep(0.1)

# status_text.text('Done!')
# st.balloons()

