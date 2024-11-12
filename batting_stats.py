# coding: utf-8

# # Using Machine Learning to find MLB Players Who were Underutilized in 2019  

# ## Data taken from off of Baseball-Reference for 2019 Statistics:
	
	# https://www.baseball-reference.com/leagues/MLB/2019-standard-batting.shtml


#Importing all of the necessary Python packages
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# ## Below is data manipulation and structuring to return a dataframe that contains all relevant information of all players who spent the entire season with only one team

# You may not need to specify the engine "openpyxl". I did so this could run on Android 
# as well as Windows and Mac
batting_stats_dataframe = pd.read_excel("batting_stats_2019.xlsx",engine='openpyxl')

batting_stats_dataframe = batting_stats_dataframe.fillna(0)


# ### Removing Traded Players (Players on more than one 1 team during the year)


traded_players = batting_stats_dataframe[batting_stats_dataframe.groupby('Name')['Name'].transform('size') > 1]


traded_players = traded_players['Name'].tolist()

stats_df = batting_stats_dataframe[batting_stats_dataframe['Name'].isin(traded_players)==False]

players = stats_df['Name'].tolist()

#Removing unnecessary Statistics
filtered_stats_df = stats_df.drop(['Rk','Name','Tm','Lg','Pos\xa0Summary'], axis=1)

stats_array = filtered_stats_df.values

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(stats_array)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

# ## k-Means Clustering of Data with Labeled Graphs

from sklearn.cluster import KMeans
#Choosing 5 clusters and assigning each player a cluster
clf = KMeans(n_clusters=5)
clf.fit(stats_array)
y_predict = clf.fit_predict(stats_array)
filtered_stats_df['Cluster'] = y_predict
principalDf['Cluster'] = y_predict

cluster_centers_df = pd.DataFrame(data=clf.cluster_centers_,columns=filtered_stats_df.columns[0:25])

#Merging Player Stats with Player Names and Cluster Identification
full_player_name_stats = pd.merge(left=stats_df[['Name','Tm','Lg','Pos\xa0Summary']], right=filtered_stats_df , left_index=True,right_index=True)

# # Second Clustering of Day-to-Day Players

#Selecting all the clusters which contain players who played AT LEAST  half of the season
#This ensures only players who are everyday hitters are selected. Pitchers and minor-league
#"callups" - ie, players without enough batting experience - will not included in the analysis
#This second clustering of the previously clustered data will determine which data point
#(players) belong to which clusters (be it All-stars, everday or undervalued...)
full_time_player_clusters = cluster_centers_df[cluster_centers_df['G']>81].index

# Creating a copy of the slice of tthe filtered stats dataframe for future use
full_time_players = filtered_stats_df[filtered_stats_df['Cluster'].isin(full_time_player_clusters)].copy()

#Creating an array based on dataframe
player_stats_array = full_time_players.values

#Creating principal components for the second round of clustering and eventual graph
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(player_stats_array)
principalDf_daily = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])


#Clustering based on the data from daily hitters
# This returns a warning. I am aware. However, the alternative will return a "null value"
clf = KMeans(n_clusters=5)
clf.fit(stats_array)
y_predict = clf.fit_predict(player_stats_array)
full_time_players['Cluster2'] = y_predict
principalDf_daily['Cluster'] = y_predict


# 2-D Plot of Day-to-day player data based on principal components
colors=["#0000FF", "#00FF00", "#FF0066","purple","orange"]
# blue, green, pink, purple, orange

x = principalDf_daily['principal component 1'].values
y = principalDf_daily['principal component 2'].values

#Assigning each point relative to the cluster it belongs
for i in range(len(x)):
    color_index = principalDf_daily['Cluster'].loc[i]
    plt.scatter(x[i], y[i], color=colors[color_index])
plt.xlabel('Princpal Component 1')
plt.ylabel('Princpal Component 2')
plt.title('2D PCA Projection of Day-To-Day Player  2019 MLB Data')
plt.savefig("everyday_clusters.png")

cluster_centers_df = pd.DataFrame(data=clf.cluster_centers_,columns=filtered_stats_df.columns[0:26])

#Merging Sats with Player names to include all data in one data object
player_name_stats = pd.merge(left=stats_df[['Name','Tm','Lg','Pos\xa0Summary']], right=full_time_players , left_index=True,right_index=True)

# Dropping average player values
player_name_stats.drop(player_name_stats.tail(1).index,inplace=True)

from matplotlib.ticker import FormatStrFormatter

colors=["#0000FF", "#00FF00", "#FF0066","purple","orange"]
# blue, green, pink, purple, orange

# Creating a 3-D plot of Games, Home Runs and Average for everyday players to view clusters
x = player_name_stats['G'].values
y = player_name_stats['BA'].values
z = player_name_stats['HR'].values
color_indices = player_name_stats['Cluster2'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.tight_layout()

ax.set_xlabel('Games')
ax.set_ylabel('Batting Average')
ax.set_zlabel('HR',rotation=270,labelpad=-5)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax.set_box_aspect(None, zoom=0.95)
ax.set_title('Games and Average Production for 2019 MLB Season')
ax.text2D(0.2, 0.97, "Similarly Colored Data Points Belong to Same Cluster", transform=ax.transAxes)

for i in range(len(x)):
    color_index = color_indices[i]
    ax.scatter(x[i], y[i], z[i],color=colors[color_index])
plt.savefig("everyday_stats.png",bbox_inches='tight')


# # Exporting to Excel

player_name_stats.to_excel("batting_clusters.xlsx")
