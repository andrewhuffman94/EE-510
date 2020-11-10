# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:15:52 2020

@author: Andrew
"""
# This code follows the UMAP tutorial found at https://umap-learn.readthedocs.io/en/latest/basic_usage.html
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from scipy.io import loadmat
from io import BytesIO
from PIL import Image
import base64
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

# Load data and define testing/training examples and labels
data = loadmat('mnistFull.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'][:,0]
y_test = data['y_test'][:,0]
target_names = np.arange(0,10)

# Perform UMAP embedding and generate matplotlib static plot
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_test.T)
plt.scatter(embedding[:,0],embedding[:,1], c=y_test, cmap="Spectral", s=5)
plt.gca().set_aspect("equal", "datalim")
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title("UMAP projection of the MNIST dataset")


# Define function to get images from data 
def embeddable_image(X):
    img_data = (255*X.T).astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((64, 64),Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

# Generate interactive plot
X = X_test.T.reshape((10000,28,28))
digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x) for x in y_test]
digits_df['image'] = list(map(embeddable_image,X))
datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(factors=[str(9),str(8),str(7),str(6),str(5),str(4),str(3),str(2),str(1),str(0)],palette=Spectral10)
plot_figure = figure(title='UMAP projection of the MNIST dataset',plot_width=600,plot_height=600,tools=('pan, wheel_zoom, reset'))
plot_figure.add_tools(HoverTool(tooltips="""<div> <div> <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/> </div> <div> <span style='font-size: 16px; color: #224499'>Digit:</span> <span style='font-size: 18px'>@digit</span> </div> </div>"""))
plot_figure.circle('x','y',source=datasource,color=dict(field='digit', transform=color_mapping),line_alpha=0.6,fill_alpha=0.6,size=4)
show(plot_figure)
