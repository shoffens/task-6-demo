# cd OneDrive\Stevens\Research\ART-002\python
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import random
import math
import numpy as np
from scipy import optimize

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

initial_MC_size = 500
x2 = [random.random() for _ in range(initial_MC_size)]
y2 = [random.random() for _ in range(initial_MC_size)]
target_color = 'red'
objs1 = {'Range (mi)', 'Cost (M$)', 'Mission Success (%)'}
objslist = ['Range (mi)', 'Cost (M$)', 'Mission Success (%)']
cannondata = pd.read_csv("missiledata_complete.csv") # MAY CAUSE PROBLEM WITH DIRECTORY
#cannondata = pd.read_csv("/home/hoffenson/mysite/missiledata_complete.csv") # FOR PYTHONANYWHERE
objs2 = cannondata.columns
objs2_noname = cannondata.columns[1:,]
################## Dash layout ###################
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # FOR HEROKU
app.title = 'Task 6 - Visualization'

app.layout = html.Div(children=[
    html.H3('Task 6: Visualization dashboard of system capability and mission success', style={
            'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),

    dcc.Tabs(id="tabs", children = [

        ###################### TAB 1 LAYOUT ########################
        dcc.Tab(label='Design explorer', children = [
        #Inputs: caliber, power, barrel length, number of cannons, smart/dumb
        #Parameters, air pressure, target distance, wind
        #Outputs: range, accuracy, system weight, projectile weight, cost, blast radius??
            html.Div([

                dcc.Markdown('**Input system attributes**'),

                html.Label('Cannon power (hp)'),
                dcc.Input(
                    id='cannon-power',
                    value=250,
                    # step=25,
                    type='number',
                    min=0,
                    max=1000,
                    # updatemode='keypress'
                ),

                html.Label('Caliber (mm)'),
                dcc.Input(
                    id='cannon-diameter',
                    value=150,
                    # step=5,
                    type='number',
                    min=5,
                    max=400
                ),

                html.Label('Barrel length (m)'),
                dcc.Input(
                    id='barrel-length',
                    value=7,
                    step=0.1,
                    type='number',
                    min=3,
                    max=14
                ),

            ], style={'width': '190px', 'display': 'inline-block','vertical-align':'top'}),

            html.Div(style={'width': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div(children=[
                dcc.Markdown('**System representation**'),
                dcc.Graph(id='basic-interactions',config={'displayModeBar': False})
            ], style={'width': '350px', 'height': '200px', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div(style={'width': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div(children=[
                dcc.Markdown('**System capabilities**'),
                dcc.Graph(id='basic-outcomes',config={'displayModeBar': False})
            ], style={'width': '375px', 'display': 'inline-block', 'vertical-align': 'top'}), #, 'height':'300px'

        ]),

        ###################### TAB 2 LAYOUT ########################
        dcc.Tab(label='Mission explorer', children = [
            html.Div([

                dcc.Markdown('**Input mission attributes here**'),

                html.Label('Monte Carlo size'),
                dcc.Input(
                    id='MC-number',
                    value=initial_MC_size,
                    step=10,
                    type='number',
                    min=10
                ),

                html.Label('Maximum target distance (miles)'),
                dcc.Input(
                    id='target-max-distance',
                    value=20,
                    step=1,
                    type='number',
                    min=0,
                    max=1000
                ),

                html.Label('Misfire probability (%)'),
                dcc.Input(
                    id='prob-misfire',
                    value=5,
                    step=1,
                    type='number',
                    min=0,
                    max=100
                ),

                html.Label('Number of cannons (inactive)'),
                dcc.Input(
                    id='num-cannons',
                    value=1,
                    step=1,
                    type='number',
                    min=1,
                    max=100
                ),
            ], style={'width': '30%', 'display': 'inline-block','vertical-align':'top'}),

            html.Div(children=[

                html.Div([
                    html.Div([
                        html.Button(id='generate-MC-button', n_clicks=1, children='New Monte Carlo simulation'),
                        html.Div(id='MC-generation-number')
                    ], style={'marginLeft': 40,'width':'45%', 'display':'inline-block'}), #, style={'textAlign': 'center'}
                    html.Div([
                        html.H3(id='percent-hit')
                    ], style={'width':'45%', 'display':'inline-block','vertical-align':'top','textAlign':'center'})
                ]),

                dcc.Graph(id='MC-sample-graph',config={'displayModeBar': False}), #,'frameMargins': 0,0,0,0

            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align':'top'}) #'marginTop': -80, #,'textAlign':'center'}
        ]),

        ###################### TAB 3 LAYOUT ########################
        dcc.Tab(label='Tradeoff explorer', children = [
            # html.Div(children = 'This is a placeholder for an interactive visualization of Pareto optimal designs')
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='pareto-xaxis',
                        options=[{'label': i, 'value': i} for i in objs1],
                        value='Range (mi)'
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='pareto-yaxis',
                        options=[{'label': i, 'value': i} for i in objs1],
                        value='Cost (M$)'
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={
                'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px 5px'}
            ),

            html.Div([
                html.Div([
                    dcc.Graph(
                        id='pareto-scatter',
                        config={'displayModeBar': False},
                        # hoverData={'points':[{'customdata':'none'}]}
                        hoverData={'points':[{'customdata':'Hover over a point'}]}
    #                    hoverData={'points': [{'customdata': 'M777A2'}]}#cannondata.Name[0]}]}  ##### PROBABLY DOESNT WORK!!!!!!!!!!!!!!!!!!!!!!!!!
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),
                html.Div(children=[
                    dcc.Graph(
                        id='pareto-point',
                        config={'displayModeBar': False},
                    )
                ], style={'width': '52%', 'display': 'inline-block', 'padding': '0 20'}),
            ]),
        ]),

        ###################### TAB 4 LAYOUT ########################
        dcc.Tab(label='Deployed systems',children=[
            html.Div([

                html.Div([
                    dcc.Dropdown(
                        id='compare-xaxis',
                        options=[{'label': i, 'value': i} for i in objs2_noname],
                        value='Range (mi)'
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='compare-yaxis',
                        options=[{'label': i, 'value': i} for i in objs2_noname],
                        value='Unit cost ($)'
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={
                'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px 5px'
            }),

            html.Div([
                dcc.Graph(
                    id='compare-scatter',
                    config={'displayModeBar': False},
                    hoverData={'points':[{'customdata':'Hover over a point'}]}
#                    hoverData={'points': [{'customdata': 'M777A2'}]}#cannondata.Name[0]}]}  ##### PROBABLY DOESNT WORK!!!!!!!!!!!!!!!!!!!!!!!!!
                )
            ], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                html.H3(id='system-design'),
                dcc.Markdown('Placeholder for system design image')
#                html.P('Placeholder for system design image')
#                dcc.Graph(id='system-design')
            ], style={'display': 'inline-block', 'width': '40%','vertical-align':'top','textAlign':'center'}),

        ])

    ], style={
        'fontFamily': 'system-ui'
    },
    content_style={
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'padding': '20px'
    },
    parent_style={
        'maxWidth': '1000px',
        'margin': '0 auto'
    }
    )
])

################## Functions to be used later ###################
def draw_cannon(cannon_diameter,cannon_power,barrel_length):
    barrelangle = 10*math.pi/180
    barrel_width = cannon_diameter/150
    box = -8  # barrel origin, in bottom-left corner
    boy = 8
    bsx2 = box + barrel_length*2*math.cos(barrelangle)
    bsy2 = boy + barrel_length*2*math.sin(barrelangle)
    bsx4 = box - barrel_width*math.sin(barrelangle)
    bsy4 = boy + barrel_width*math.cos(barrelangle)
    bsx3 = bsx4 + barrel_length*2*math.cos(barrelangle)
    bsy3 = bsy4 + barrel_length*2*math.sin(barrelangle)
    power_length = cannon_power*0.00533333333 + 8/3
    power_width = cannon_power*0.002+2.5
    pox = box - (barrel_width/2-power_width/2)*math.sin(barrelangle)
    poy = boy + (barrel_width/2-power_width/2)*math.cos(barrelangle)
    psx2 = pox + power_length*math.cos(barrelangle)
    psy2 = poy + power_length*math.sin(barrelangle)
    psx4 = pox - power_width*math.sin(barrelangle)
    psy4 = poy + power_width*math.cos(barrelangle)
    psx3 = psx4 + power_length*math.cos(barrelangle)
    psy3 = psy4 + power_length*math.sin(barrelangle)
    barrel_string = 'M {} {} L {} {} L {} {} L {} {} Z'.format(box,boy,bsx2,bsy2,bsx3,bsy3,bsx4,bsy4)
    power_string = 'M {} {} L {} {} L {} {} L {} {} Z'.format(pox,poy,psx2,psy2,psx3,psy3,psx4,psy4)
    return [barrel_string,power_string]
    # return ['M {} {} L {} {} L {} {} L {} {} Z'.format(box,boy,bsx2,bsy2,bsx3,bsy3,bsx4,bsy4), #barrel_string
    #         'M {} {} L {} {} L {} {} L {} {} Z'.format(pox,poy,psx2,psy2,psx3,psy3,psx4,psy4)] #power_string

def mission_success(cannon_diameter,cannon_power,max_dist,prob_misfire,MC_size):
    maxspeed = math.sqrt(2*cannon_power/(cannon_diameter/50)) # ft/s; assumes 1s of acceleration, 0.2 lb/cm of diameter
    x2i = []
    y2i = []
    x2h = []
    y2h = []
    x2m = []
    y2m = []
    hit = np.zeros(MC_size)
    for i in range(MC_size):
        x2i = x2[i]*max_dist*2 - max_dist
        y2i = y2[i]*max_dist*2 - max_dist
        target_distance = math.sqrt(x2i**2 + y2i**2)
        init_speed_reqd = math.sqrt((9.8*target_distance)/(math.sin(math.pi/2))) ## =98 for distance =10
        if (init_speed_reqd <= maxspeed and random.random() >= prob_misfire/100):
            hit[i] = 1
            x2h.append(x2i)
            y2h.append(y2i)
        else:
            hit[i] = 0
            x2m.append(x2i)
            y2m.append(y2i)
    prob_hit = sum(hit)/MC_size * 100
    return [prob_hit,x2h,y2h,x2m,y2m]

################## Dash update logic for Tab 1 ##################
@app.callback([dash.dependencies.Output('basic-interactions', 'figure'),
    dash.dependencies.Output('basic-outcomes','figure')],
    [dash.dependencies.Input('cannon-diameter','value'),
    dash.dependencies.Input('cannon-power', 'value'),
    dash.dependencies.Input('barrel-length', 'value')])
def update_graph(cannon_diameter,cannon_power,barrel_length):
    global sysdict, syslabels, sysvalues
    cannondiametermarker = cannon_diameter/6
    cannonpowermarker = cannon_power/100
    # Kinetic energy = (1/2)*m*v^2 -> v = sqrt(2*KE/m)
    maxspeed = math.sqrt(2*cannon_power/(cannon_diameter/50)) # ft/s; assumes 1s of acceleration, 0.2 lb/cm of diameter
    maxdistance = (maxspeed**2)*math.sin(math.pi/2)/9.8
    # v_ftpersec = math.sqrt(2*cannon_power/(cannon_diameter/50)) # ft/s; assumes 1s of acceleration, 0.2 lb/cm of diameter
    # v = v_ftpersec/3.28084
    # # maxdistance = -343.6*v**5 + 442.1*v**4 + 1075*v**3 - 2512*v**2 + 4417*v + 11480
    # maxdistance = -1.471e-10*v**5 + 4.35e-07*v**4 - 0.0004653*v**3 + 0.1992*v**2 - 6.019*v + 34.03
    # maxdistance_m = -1.4706E-10*v**5 + 4.3500E-07*v**4 - 4.6531E-04*v**3 + 1.9917E-01*v**2 - 6.0187E+00*v + 3.4027E+01
    # maxdistance = maxdistance_m/1609.34
    # print(maxdistance)
    projectile_color = 'blue'
    accuracy = 10*cannon_diameter/(barrel_length*(cannon_power+0.001))
    sysmass = cannon_power*cannon_diameter*barrel_length/25000
#    projectilemass = 2.8573*math.exp(0.0238*cannon_diameter)
    projectilemass = 9.9984838006*10**(-7)*cannon_diameter**3.7144157509
    unitcost = cannon_diameter*(barrel_length**2)*cannon_power/1000000

    norm_dist = maxdistance/2040 #min=0, max=2040
    norm_acc = (accuracy - 0.00357)/(10-0.00357) #min/best=0.00357, max/worst=10 (actually 1330000); use log!
    norm_smass = sysmass/224 #max/worst = 224, min/best = 0
    norm_pmass = (projectilemass - 0.000395)/(4620-0.000395) #max/worst = 4620, min/best = 0 (actually 0.000395); use log!
    norm_ucost = unitcost/78.4 #max/worst = 78.4M, min/best = 0

    syslabels = ['Range (mi)', 'Caliber (mm)', 'System weight (lb)', 'Length (m)', 'Year', 'Unit cost ($)', 'Projectile weight (lb)']
    sysvalues = [maxdistance, cannon_diameter, sysmass*2000, barrel_length, 2019, unitcost*1000000, projectilemass]
    sysdict = {'Range (mi)':maxdistance , 'Caliber (mm)':cannon_diameter , 'System weight (lb)':sysmass*2000 , 'Length (m)':barrel_length , 'Year':2019 , 'Unit cost ($)':unitcost*1000000 , 'Projectile weight (lb)':projectilemass}

    # DRAWING VARIABLES:
    wdia = 1.3  # Wheel diameter
    barrel_string = draw_cannon(cannon_diameter,cannon_power,barrel_length)[0]
    power_string = draw_cannon(cannon_diameter,cannon_power,barrel_length)[1]

    # BAR CHART PARAMETERS:
    attslist = ['Unit cost', 'Projectile mass', 'System mass', 'Accuracy', 'Range']
    barcolors = ['indianred',] * 5
    barcolors[4] = 'mediumaquamarine' #'palegreen'

    return [
        # System drawing:
        {'data': [],
        'layout': {
            'xaxis': {
                'range': [-9, 20],
                'showticklabels': False,
                'showgrid': False,
                'zeroline': False,
            },
            'yaxis': {
                'range': [3, 16],
                'showticklabels': False,
                'showgrid': False,
                'zeroline': False,
            },
            'shapes': [
                {'type': 'rect', 'x0': -8, 'y0': 5, 'x1': 2, 'y1': 9, 'fillcolor': 'white'},
                {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': -6-wdia, 'y0': 5-wdia, 'x1': -6+wdia, 'y1': 5+wdia, 'fillcolor': 'black'},
                {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': 0-wdia, 'y0': 5-wdia, 'x1': 0+wdia, 'y1': 5+wdia, 'fillcolor': 'black'},
                {'type': 'path', 'path':power_string, 'fillcolor': 'black'},   # Base
                {'type': 'path', 'path':barrel_string, 'fillcolor': 'black'}   # Barrel
            ],
            'height': 160,
            'margin': {
                    'l': 0, 'b': 0, 't': 0, 'r': 0
            }
        }},
        # Outcome bar charts:
        {'data': [go.Bar(
            x=[norm_ucost,norm_pmass,norm_smass,norm_acc,norm_dist],
            y=['Unit cost', 'Projectile mass', 'System mass', 'Accuracy', 'Range'],
            # hovertext = ['${} M'.format(float('%.3g' % unitcost)),'b','c','d','e'],
            # width = [0.7,]*5,
            hoverinfo="none",
            text=['${} M'.format(float('%.3g' % unitcost)),
                '{} lb'.format(float('%.3g' % projectilemass)),
                '{} tons'.format(float('%.3g' % sysmass)),
                '{} mm at 1km'.format(float('%.3g' % accuracy)),
                '{} miles'.format(float('%.3g' % maxdistance))],
            textposition='auto',
            textfont=dict(
                size=16,
                color='black'
            ),
            marker_color = barcolors,
            marker_line_color='black',
            marker_line_width=0.5,
            orientation='h')],
        'layout': {
            'xaxis': dict(
                type='log',
                range = [-4,0],
                showticklabels = False,
                #'showgrid': False,
                #'zeroline': False,
            ),
            'yaxis': dict(
                zeroline = True
            ),
            'height': 350,
            'margin': {
                    'l': 100, 'b': 100, 't': 0, 'r': 0
            }
        }}
    ]

################## Dash update logic for Tab 2 ##################
@app.callback(dash.dependencies.Output('MC-generation-number', 'children'),
    [dash.dependencies.Input('generate-MC-button', 'n_clicks'),
    dash.dependencies.Input('MC-number', 'value')])
def update_output(n_clicks,MC_size):
    global x2,y2
    x2 = [random.random() for _ in range(MC_size)]
    y2 = [random.random() for _ in range(MC_size)]
    return u'Random generation: {}, size: {}'.format(n_clicks,MC_size)

@app.callback([dash.dependencies.Output('percent-hit', 'children'),
    dash.dependencies.Output('MC-sample-graph', 'figure')],
    [dash.dependencies.Input('MC-number','value'),
    dash.dependencies.Input('target-max-distance', 'value'),
    dash.dependencies.Input('prob-misfire', 'value'),
    dash.dependencies.Input('cannon-diameter','value'),
    dash.dependencies.Input('cannon-power', 'value'),
    dash.dependencies.Input('generate-MC-button', 'n_clicks')])
def update_MC_graph(MC_size,max_dist,prob_misfire,cannon_diameter,cannon_power,nclicks):
    cannondiametermarker = cannon_diameter/12
    cannonpowermarker = cannon_power/200

    mission = mission_success(cannon_diameter,cannon_power,max_dist,prob_misfire,MC_size)
    prob_hit = mission[0]
    x2h = mission[1]
    y2h = mission[2]
    x2m = mission[3]
    y2m = mission[4]

    cannon_color = 'blue'

    return [
        ['{}% Hits'.format(float('%.3g' % prob_hit))],
        {'data': [
            {
                'x': [0],
                'y': [0],
                'name': 'Target',
                'mode': 'markers',
                'marker': {
                   'size': 12,
                   'color': 'white',
                   'line': {'color': 'red', 'width': 2}
                },
               'hovertext': 'Target',
               'hoverinfo': 'text',
            },
            {
                'x': [0],
                'y': [0],
                'name': 'Target',
                'mode': 'markers',
                'hoverinfo': 'none',
                'marker': {
                    'size': 4,
                    'color': 'white',
                    'line': {'color': 'red', 'width': 2}
                },
                'showlegend': False
            },
            {
               'x': [i for i in x2h],
               'y': [i for i in y2h],
               'text': ['Cannon (Hit)'],
               'hovertext': 'Hit!',
               'hoverinfo': 'text',
               'name': 'Hits',
               'mode': 'markers',
               'marker': {'size': 8,'color':cannon_color,'opacity':0.65}
            },
            {
              'x': [i for i in x2m],
              'y': [i for i in y2m],
              'text': ['Cannon (Miss)'],
              'name': 'Misses',
             'hovertext': 'Miss!',
             'hoverinfo': 'text',
              'mode': 'markers',
              'marker': {'size': 8,'color':cannon_color,'opacity':0.2}
            },
        ],
        'layout': go.Layout(
            xaxis = {'showgrid':False,'zeroline':False,'showticklabels':False,'range':[-max_dist,max_dist],'showline':True,'mirror':True},
            yaxis = {'showgrid':False,'zeroline':False,'showticklabels':False,'range':[-max_dist,max_dist],'showline':True,'mirror':True},
#            height=500,
#            width=600,
            margin=go.layout.Margin(l=40, r=40, b=30, t=30, pad=0),
            hovermode='closest'
            ),
            # go.layout.Image(
            #     source = "Target.jpg",
            #     xref = 0,
            #     yref = 0,
            #     sizex = 2,
            #     sizey = 2,
            #     sizing = "stretch",
            #     layer = "above"
            # )
        }
    ]

################## Dash update logic for Tab 3 ##################
@app.callback(
    dash.dependencies.Output('pareto-scatter', 'figure'),
    [dash.dependencies.Input('pareto-xaxis', 'value'),
     dash.dependencies.Input('pareto-yaxis','value'),
     dash.dependencies.Input('cannon-diameter','value'),
     dash.dependencies.Input('cannon-power', 'value'),
     dash.dependencies.Input('barrel-length', 'value'),
     dash.dependencies.Input('target-max-distance', 'value'),
     dash.dependencies.Input('prob-misfire', 'value'),
     dash.dependencies.Input('MC-number','value')])
def update_graph(xaxis_pareto_id, yaxis_pareto_id,cannon_diameter,cannon_power,barrel_length,max_dist,prob_misfire,MC_size):
    global xopt_diam,xopt_pow,xopt_len
    # global xobj,yobj
    # xobj = objslist.index(xaxis_pareto_id)
    # yobj = objslist.index(yaxis_pareto_id)
    def obj_range(cannon_diameter,cannon_power):                # Ranges 0-2041
        maxspeed = math.sqrt(2*cannon_power/(cannon_diameter/50)) # ft/s; assumes 1s of acceleration, 0.2 lb/cm of diameter
        return (maxspeed**2)*math.sin(math.pi/2)/9.8
    def obj_cost(cannon_diameter,cannon_power,barrel_length):   #Ranges 0 - 78.4
        return cannon_diameter*(barrel_length**2)*cannon_power/1000000
    def obj_successrate(cannon_diameter,cannon_power,max_dist,prob_misfire,MC_size):
        mission = mission_success(cannon_diameter,cannon_power,max_dist,prob_misfire,MC_size)
        return mission[0]
    numparetos = 7
    paretox = []    # Range
    paretoy = []    # Cost
    xopt_diam = []
    xopt_pow = []
    xopt_len = []
    # Vars = {cannon_diameter,cannon_power,barrel_length}
    x0 = [150,250,7]
    ulb = ((50,400),(25,1000),(3,14))
    for i in range(numparetos):
        # funlist = ['obj_range(x[0],x[1])','obj_cost(x[0],x[1],x[2])','obj_successrate(x[0],x[1],20,5,500)']
        def optcon(x):
            cannon_diameter = x[0]
            cannon_power = x[1]
            barrel_length = x[2]
            # return eval(funlist[xobj])
            if xaxis_pareto_id == 'Range (mi)':
                return obj_range(cannon_diameter,cannon_power)
            elif xaxis_pareto_id == 'Cost (M$)':
                return obj_cost(cannon_diameter,cannon_power,barrel_length)
            elif xaxis_pareto_id == 'Mission Success (%)':
                return obj_successrate(cannon_diameter,cannon_power,20,5,500)
        def optobj(x):
            cannon_diameter = x[0]
            cannon_power = x[1]
            barrel_length = x[2]
            return obj_cost(cannon_diameter,cannon_power,barrel_length)
            # return -obj_successrate(cannon_diameter,cannon_power,max_dist,prob_misfire)
#        coneps = (i/(numparetos-1))*2041
        # constr = optimize.NonlinearConstraint(optcon,coneps,10000)
        if xaxis_pareto_id == 'Range (mi)':
            coneps = 1+(i/(numparetos-1))*406
            constr = optimize.NonlinearConstraint(optcon,coneps,500)
        elif xaxis_pareto_id == 'Cost (M$)':
            coneps = (i/(numparetos-1))*0.5
            constr = optimize.NonlinearConstraint(optcon,0,coneps)
        elif xaxis_pareto_id == 'Mission Success (%)':
            coneps = (i/(numparetos-1))*100
            constr = optimize.NonlinearConstraint(optcon,coneps,101)
        optresult = optimize.minimize(optobj, x0, method='SLSQP', bounds=ulb, constraints=constr) #, options={'eps':0.05})
        xmin = optresult.x
        xopt_diam.append(xmin[0])
        xopt_pow.append(xmin[1])
        xopt_len.append(xmin[2])
        if xaxis_pareto_id == 'Range (mi)':
            paretox.append(obj_range(xmin[0],xmin[1]))
        elif xaxis_pareto_id == 'Cost (M$)':
            paretox.append(obj_cost(xmin[0],xmin[1],xmin[2]))
        elif xaxis_pareto_id == 'Mission Success (%)':
            paretox.append(obj_successrate(xmin[0],xmin[1],20,5,500))
        # paretox.append(obj_range(xmin[0],xmin[1]))
        paretoy.append(obj_cost(xmin[0],xmin[1],xmin[2]))
    return {
        'data': [
                   {
                       'x': paretox,
                       'y': paretoy,
                       'text': ['Pareto point'],
                       'name': 'Pareto point',
                       'customdata': [_ for _ in range(numparetos)],
                       'mode': 'markers',
                       'marker':{'size': 15,'color':'red','opacity':0.5}
                   }
               ],
        'layout': go.Layout(
           xaxis={'title': xaxis_pareto_id,'type': 'linear'},
           yaxis={'title': yaxis_pareto_id,'type': 'linear'},
           hovermode='closest'
        )
    }

@app.callback(
    dash.dependencies.Output('pareto-point', 'figure'),
    [dash.dependencies.Input('pareto-scatter', 'hoverData')])
def update_pareto_drawing(hoverData):
    pareto_id = hoverData['points'][0]['customdata']
    # if pareto_id == 'none':
    if isinstance(pareto_id,str)==True:
        return {
            'data': [go.Scatter(
                x=[1],
                y=[1],
                mode="markers+text",
                name="Markers and Text",
                text=["Hover over a point to the left"],
                textposition="top center",
                marker={
                    'size': 15,
                    'opacity': 0,
                    'line': {'width': 0, 'color': 'white'}
                },
                textfont=dict(
                    size=18
                )
            )],
            'layout': {
                'xaxis': {
                    'range': [-5,5],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                },

                'yaxis': {
                    'range': [-5, 5],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                }
            }
        }
    else:
        cannon_diameter = xopt_diam[pareto_id]
        cannon_power = xopt_pow[pareto_id]
        barrel_length = xopt_len[pareto_id]
        # DRAWING VARIABLES:
        wdia = 1.3  # Wheel diameter
        barrel_string = draw_cannon(cannon_diameter,cannon_power,barrel_length)[0]
        power_string = draw_cannon(cannon_diameter,cannon_power,barrel_length)[1]
        return {'data': [],
            'layout': {
                # 'height': 100,
                # 'width': 100,

                'xaxis': {
                    'range': [-10, 20],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                },

                'yaxis': {
                    'range': [-10, 15],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                },

                'shapes': [
                    {'type': 'rect', 'x0': -8, 'y0': 5, 'x1': 2, 'y1': 9, 'fillcolor': 'white'},
                    {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': -6-wdia, 'y0': 5-wdia, 'x1': -6+wdia, 'y1': 5+wdia, 'fillcolor': 'black'},
                    {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': 0-wdia, 'y0': 5-wdia, 'x1': 0+wdia, 'y1': 5+wdia, 'fillcolor': 'black'},
                    {'type': 'path', 'path':power_string, 'fillcolor': 'black'},   # Base
                    {'type': 'path', 'path':barrel_string, 'fillcolor': 'black'}   # Barrel
            ]}}

################## Dash update logic for Tab 4 ##################
@app.callback(
    dash.dependencies.Output('compare-scatter', 'figure'),
    [dash.dependencies.Input('compare-xaxis', 'value'),
     dash.dependencies.Input('compare-yaxis','value')])
def update_graph(xaxis_column_id, yaxis_column_id):
#    idstrings = np.array(["%.0f" % _ for _ in objs2[:,0]])
#    scatterlabels=['Pareto ' + i for i in idstrings]
    scatterlabels=cannondata.get("Name")
    return {
        'data': [go.Scatter(
            x=cannondata.get(xaxis_column_id),
            y=cannondata.get(yaxis_column_id),
            text=scatterlabels,
            customdata=scatterlabels,
            name='Existing systems',
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        ),
            {
              'x':[sysdict.get(xaxis_column_id)],
              'y':[sysdict.get(yaxis_column_id)],
              'text':'New design',
              'customdata':['New design'],
              'name':'New system',
              'mode':'markers',
              'marker':{'size': 15,'color':'red','opacity':0.5}
            }
        ],
        'layout': go.Layout(
            xaxis={'title': xaxis_column_id,'type': 'linear'},
            yaxis={'title': yaxis_column_id,'type': 'linear'},
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }

@app.callback(
    dash.dependencies.Output('system-design', 'children'),
    [dash.dependencies.Input('compare-scatter', 'hoverData')])
def update_drawing(hoverData):
    design_id = hoverData['points'][0]['customdata']       #hoverData['points'][0]['customdata']
#    design_id = hoverData['points'][:]['customdata']
    return '{}'.format(design_id)

# REMOVE THE BELOW WHEN UPLOADING TO PYTHONANYWHERE:
if __name__ == '__main__':
    app.run_server(debug=True)
