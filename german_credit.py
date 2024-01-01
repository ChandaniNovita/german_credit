import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('german_credit_data.csv')
numeric = ['Age', 'Job', 'Credit amount', 'Duration']
categorical = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')
data = df.copy()
# Define encoder here
encoder = LabelEncoder()
for label in categorical:
    data[label] = encoder.fit_transform(data[label])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(X_scaled, columns=data.columns)

X = data
st.header("isi dataset")
st.write(data)

clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2, 10, 3, 1)

def k_means(n_clust):
    X_copy = X.copy()
    kmean = KMeans(n_clusters=n_clust).fit(X_copy)
    X_copy['Labels'] = kmean.labels_
    return X_copy



st.header('Cluster Plot')
result_df = k_means(clust)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(result_df['Age'], result_df['Job'], result_df['Credit amount'], c=result_df['Labels'], cmap='viridis')

ax.set_xlabel('Age')
ax.set_ylabel('Job')
ax.set_zlabel('Credit amount')

legend = ax.legend(*scatter.legend_elements(), title='Labels')
ax.add_artist(legend)

st.pyplot(fig)
st.write(result_df)
