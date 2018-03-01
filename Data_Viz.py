
# coding: utf-8

# # DATA VISULIZATION

# In[ ]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
from haversine import haversine


# Reading the data from data_tohoku_norm_transpose.csv

# In[2]:

array_vals = pd.read_csv("Downloads/data_tohoku_norm_transpose.csv", header=None)
array_vals.head()


# In[3]:

v = pd.date_range("2:46PM", "6:46PM", freq="1s")
v -= v[0]
array_vals["time"] = v
array_vals.set_index("time", inplace=True)


# Reading the data of Latitudes and Longitudes from location.txt

# In[4]:

locations = pd.read_csv("Downloads/location.txt",
                        delimiter="\t", names =["longitude", "latitude", "a", "b"])
del locations["a"], locations["b"]


# Creating a dictionary 'd' for interactivity between the Map and slider

# In[5]:

d={}
#for i in range(5):
#    d[i]=array_vals.iloc[i].tolist()
z=range(14401)
a=[str(i) for i in z]
b=[]
for i in range(14401):
    b.append(array_vals.iloc[i].tolist())
d=dict(zip(a,b))


# Creating a dictionary 'd1' for interactivity between Map and Line graph

# In[6]:

c={}
j=[]
h=[str(i) for i in range(438)]
#lon=locations["longitude"].tolist()
#u=[str(i) for i in lon]
for i in range(438):
    j.append(array_vals[i].tolist())
d1=dict(zip(h,j))


# Creating a list of latitudes and longitudes

# In[7]:

lon=locations["longitude"].tolist()
lat=locations["latitude"].tolist()


# Creating a list which has distance of each station from the point of origin(Tohoku)

# In[8]:

tohoku = (38.296, 142.4)
stations=(lat[0], lon[0]) 
#haversine(lyon, paris)
distance_in_miles=haversine(tohoku, stations, miles=True)
distance_in_miles
dis_list=[]
for i in range(len(lat)):
    stations=(lat[i], lon[i]) 
    distance_in_miles=haversine(tohoku, stations, miles=True)
    dis_list.append(distance_in_miles)


# In[9]:

df=array_vals
df.loc[0] = dis_list
df=df.T
df=df.sort_values(by=[0])
del df[0]
array_vals_new=df.T
array_vals_new.head()


# Creating a new dataframe 'gt' which has data sorted on the basis of distance from the origin

# In[10]:

spec_data=array_vals_new
spec_data.loc[0] = range(438)
gt=spec_data.T
gt=gt.set_index(0)
gt=gt.T


# Again reading the data into array_vals

# In[11]:

array_vals = pd.read_csv("Downloads/data_tohoku_norm_transpose.csv", header=None)
array_vals.head()


# Loding all the modules for creating the interactive visulization using Bokeh

# In[12]:

from ipywidgets import interact
import numpy as np
import bokeh
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()


# Creating an arrray for the generation of Spectrogram

# In[13]:

import numpy as np
h=[]
for i in range(438):
    h.append(gt[i].tolist())
h=np.asarray(h)


# Creating the final visualization using Bokeh

# In[ ]:

from bokeh.sampledata import us_states
from bokeh.plotting import *
from bokeh.models import mappers,CustomJS,ColumnDataSource,HoverTool, TapTool,Slider, Toggle
from bokeh.palettes import Reds6 as palette
from bokeh.resources import CDN
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, widgetbox
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models.mappers import LogColorMapper

us_states = us_states.data.copy()

del us_states["HI"]
del us_states["AK"]

# separate latitude and longitude points for the borders
#   of the states.
state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

TOOLS = "pan,wheel_zoom,reset,hover,save,tap"

# init figure
p = figure(title="Plotting USarray stations", 
           toolbar_location="left",  x_axis_label='Longitude', y_axis_label='Latitude', tools="lasso_select",plot_width=750, plot_height=500)

# Draw state lines
p.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color="#884444", line_width=1.5)


# Now group these values together into a lists of x (longitude) and y (latitude)
x = locations["longitude"].tolist()
y = locations["latitude"].tolist()
custom_colors = ['#f2f2f2', '#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26']
color_mapper = LogColorMapper(palette=custom_colors)    

data = dict(x=x, y=y,
            rate=array_vals.iloc[0].tolist(),**t)
s1 = ColumnDataSource(data)

#creating stations over the map of United States
p.circle('x', 'y', source=s1,
          fill_color={'field': 'rate', 'transform': color_mapper},
          fill_alpha=2,size=7)

#Creating the spectrogram
sp = figure(title="Spectrogram", x_range=(0, 14401), y_range=(0, 438),x_axis_label='Time in seconds', y_axis_label='Stations by distance from origin',plot_width=450, plot_height=250)

sp.image(image=[h], x=0, y=0, dw=14401, dh=438, palette="Spectral11")

#hover = p.select_one(HoverTool)
#hover.point_policy = "follow_mouse"
#hover.tooltips = [
 #   ("(Lon, Lat)", "($x, $y)"),
#]

g=[i for i in range(14401)]

s2=ColumnDataSource(data=dict(z=array_vals[0].tolist(),g=g,**d1))

#Creating the Line graph
n = figure(title="Line graph for selected station",x_axis_label='Time in seconds', y_axis_label='Frequency',
         plot_width=450, plot_height=250)
n.line('g','z',source=s2)


s1.callback = CustomJS(args=dict(s2=s2),code="""
var d2 = s2.data;
s2.data['z']=d2[cb_obj.selected['1d'].indices]
console.log(d2['z'])
s2.trigger('change');
console.log('Tap event occured at x-position: ' + cb_obj.selected['1d'].indices)
""")
#Creating a slider
slider = Slider(start=0, end=14400, value=0, step=1, title="Time slider in seconds")
def update(s1=s1,slider=slider, window=None):
    data = s1.data
    v = cb_obj.value
    data['rate'] = data[v]
    s1.trigger('change')

slider.js_on_change('value', CustomJS.from_py_func(update))
g1=(column(p,widgetbox(slider),))
g2= column(sp,n)
layout = row(g1, g2)
show(layout)


# In[ ]:



