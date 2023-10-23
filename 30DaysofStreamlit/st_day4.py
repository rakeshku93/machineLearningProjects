# import libarries
import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as plt_exp
import plotly.graph_objects as grp_obj

# load data
df_agg_videos = pd.read_csv("./inputs/Aggregated_Metrics_By_Video.csv")
df_agg_country = pd.read_csv("./inputs/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv")
df_comments = pd.read_csv("./inputs/All_Comments_Final.csv")
df_videos_perf = pd.read_csv("./inputs/Video_Performance_Over_Time.csv")



