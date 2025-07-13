#Fuel prediction dashboard system
#import libraries
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_show_results 
from sklearn.neighbors import KNeighborsRegressor
#import statsmodels.formula.api as smf 
from sklearn.model_selection import learning_curve,LearningCurveDisplay
#XGBoost
from xgboost import XGBRegressor
#warnings
import warnings
warnings.filterwarnings('ignore')

#For Plotly
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

import dash
from dash import callback
from dash import dcc
from dash import html

from dash.dependencies import Input,Output

options = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(), "Lasso Regression": Lasso(),"ElasticNet": ElasticNet(),"XGBRegressor" : XGBRegressor()}

app = dash.Dash()


# Part - 1: EDA
#---------------------------------------------------------------------
#import and clean/pre-process data

#df = pd.read_csv(r'C:\Users\st_ba\Downloads\auto-mpg.csv',na_values = ['?'])
df = pd.read_csv(r'data/auto-mpg.csv',na_values = ['?'])

data = df.copy()
data["origin"] = data["origin"].astype(str)


data['origin'] = data['origin'].apply(lambda x: 'USA' if x == '1' else ('Japan' if x == '2' else 'Europe'))


print(data.head())
df = df.drop(columns = ['car name'])
 

#print(df.head())
#print(df.info())

#handle missing values 
#First replace '?' under horsepower with NaN
df['horsepower'].replace('?',np.nan,inplace = True)
df['horsepower'] = df['horsepower'].astype(float)   #convert to float

#fill the missing values in the horsepower column with horsepower mean
df['horsepower'] = df['horsepower'].fillna(data['horsepower'].mean())
df.horsepower.isnull().sum()
#print("Origin:",df["origin"])
#df.describe().


X = df.drop(['mpg'], axis=1)
Y = df['mpg']
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = np.log1p(df['mpg'])


css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"]
app = dash.Dash(name="Auto-MPG EDA and Regression Dashboard", external_stylesheets=css)

# ---------------------------------------------------------------
#Create charts

def create_scatterplot(col1 = "horsepower",col2 = "mpg"):
    fig = px.scatter(data,x = col1,y = col2,hover_data = "car name",color = "origin",)
    fig.update_layout(paper_bgcolor="#e5ecf6", height=600)
    return fig

    
def create_boxplot(col_name = "horsepower"):
    fig = px.box(df,y = df[col_name])
    fig.update_layout(paper_bgcolor="#e5ecf6", height=600)
    return fig

def create_histogram(col = "horsepower"):
    fig = px.histogram(data,x = data[col],color = 'origin')
    fig.update_layout(paper_bgcolor="#e5ecf6", height=600)
    return fig
    
def create_violinplot(vln_col = "horsepower"):
    fig = px.violin(df, y = df[vln_col],box = True)
    fig.update_layout(paper_bgcolor="#e5ecf6", height=600)
    return fig

def corr_heatmap(dataframe):
    cor_mat = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z = cor_mat,
        x = cols,
        y = cols,
        colorscale='Plasma'))
    fig.update_layout(xaxis_title = 'X-axis', yaxis_title = 'Y-axis')
    return fig


def train_show_results(X, Y, model, split_share = 0.2):
    print(f"Training using {model}")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_share)
    m = model.fit(X_train, Y_train)
    preds = m.predict(X_test)
    mse = mean_squared_error(Y_test, preds)
    return Y_test,preds
    
    
    

def reg(model_name = LinearRegression()):
    Y_test,preds = train_show_results(X,Y,model_name)
    train_size_abs, train_scores, test_scores = learning_curve(model_name, X, Y)
    fig = go.Figure()
    #fig = px.scatter(x = Y_test,y = preds,trendline = 'ols',trendline_color_override = 'midnightblue',
    #                 title = "Multi-variate regression")
    fig.add_trace(go.Scatter(x=Y_test, y=preds, mode='markers',
                             marker=dict(color='midnightblue'),
                             name='Predictions'))
    fig.update_layout(paper_bgcolor="#e5ecf6", height=600)
    return fig
    
   
        

        
#-----------------------------------------------------------------
#Widgets
cols = df.columns
#print(cols,type(cols))
scat_col1 = dcc.Dropdown(id="scat_col1", options=cols, value="horsepower",clearable = True)
scat_col2 = dcc.Dropdown(id="scat_col2", options=cols, value="mpg",clearable = True)

#line_col1 = dcc.Dropdown(id = "line_col1",options = cols, value = "horsepower",clearable = True)
#line_col2 = dcc.Dropdown(id = "line_col2",options = cols,value = "mpg",clearable = True)

box_col = dcc.Dropdown(id = "box_col",options = cols,value = "horsepower",clearable = True)

hist_col = dcc.Dropdown(id = "hist_col",options = cols,value = "horsepower",clearable = True)

vln_col = dcc.Dropdown(id = "vln_col",options = cols,value = "horsepower",clearable = True)

cor_map = html.Button('Submit', id = "cor_map", n_clicks = 0)

lin_reg = dcc.Dropdown(id = 'dropdown', options = ["Linear Regression", "Ridge Regression", "Lasso Regression","ElasticNet","XGBRegressor"], value = 'Linear Regression', clearable = True)

# ---------------------------------------------------------------
#App Layout

app.layout = html.Div([
    html.Div([
        html.H1("Auto-MPG EDA and Regression Dashboard"),
        html.Br(),
        dcc.Tabs([
            dcc.Tab([html.Br(), "x:",scat_col1 , "y:",scat_col2, html.Br(),
                     dcc.Graph(id = "scatter_plot")], label="Scatter Plot"),
            #dcc.Tab([html.Br(), "x:",line_col1 , "y:",line_col2, html.Br(),
                     #dcc.Graph(id = "line_charts")], label="Line Charts"),
            dcc.Tab([html.Br(), "x:",box_col, html.Br(),
                     dcc.Graph(id = "box_plot")], label="Box Plot"),
            dcc.Tab([html.Br(), "x:",hist_col, html.Br(),
                     dcc.Graph(id = "hist_plot")], label="Histogram"),
            dcc.Tab([html.Br(), "x:",vln_col, html.Br(),
                     dcc.Graph(id = "vln_plot")], label="Violin Plot"),
            dcc.Tab([html.Br(),"Heat map",html.Br(),cor_map, html.Br(),
                     dcc.Graph(id = "cor_mat")], label="Correlation Matrix"),
            dcc.Tab([html.Br(),"Multi-variate Regression",html.Br(),lin_reg, html.Br(),
                     dcc.Graph(id = "lin_regr")], label="Linear Regression"),
            
        ])
    ], className="col-8 mx-auto"),
], style={"background-color": "#e5ecf6", "height": "100vh"})


# ---------------------------------------------------------------
#Callbacks
@callback(Output("scatter_plot", "figure"), [Input("scat_col1", "value"), Input("scat_col2", "value"),])
def update_scatterplot(scat_col1, scat_col2):
    return create_scatterplot(scat_col1, scat_col2)

'''@callback(Output("line_charts", "figure"), [Input("line_col1", "value"), Input("line_col2", "value"),])
def update_linecharts(line_col1, line_col2):
    return create_linecharts(line_col1, line_col2)'''

@callback(Output("box_plot", "figure"), [Input("box_col", "value"),])
def update_boxplot(box_col):
    return create_boxplot(box_col)

@callback(Output("hist_plot", "figure"), [Input("hist_col", "value"),])
def update_histogram(hist_col):
    return create_histogram(hist_col)

@callback(Output("vln_plot", "figure"), [Input("vln_col", "value"),])
def update_vlnplot(vln_col):
    return create_violinplot(vln_col)

@callback(
    Output('cor_mat', 'figure'),
    Input('cor_map', 'n_clicks'),
    prevent_initial_call=True
)
def update_output(n_clicks):
    return corr_heatmap(n_clicks)

@callback(Output("lin_regr", "figure"), [Input("dropdown", "value"),])
def update_linreg(name):
    model = options[name]
    print(model)
    return reg(model)

# ---------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug = True)

