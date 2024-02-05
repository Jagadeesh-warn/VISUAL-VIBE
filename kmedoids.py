import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn_extra.cluster import KMedoids
import os
import pickle
from skimage import io


# CLUSTERING (KMEDOIDS)

# cluster photos in df using Kmedoids algorithms
def kmedoids_clustering(df):
    # isolate visual word features
    df_vw = df.loc[:, ~df.columns.isin(['photo_name', 'blog_name'])]

    if not os.path.exists('models/kmedoids_bovw.pkl'):
        # find the best value for k (number of clusters)
        sse = []
        for k in range(2, 11):
            kmedoids = KMedoids(n_clusters=k, random_state=0)
            kmedoids.fit(df_vw)
            sse.append(kmedoids.inertia_)

        # select the best k from sse curve
        kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
        print("Number of cluster: " + str(kl.elbow))

        # fit the model with the best k value
        kmedoids = KMedoids(n_clusters=kl.elbow, random_state=1)
        kmedoids.fit(df_vw)

        # elbow plot
        elbow_plot(sse, kl.elbow)

        # save model
        os.makedirs('models', exist_ok=True)
        with open("models/kmedoids_bovw.pkl", 'wb') as out:
            pickle.dump({'kmedoids': kmedoids}, out)
        print("kmedoids bovw model saved")
    else:
        with open("models/kmedoids_bovw.pkl", 'rb') as inp:
            data = pickle.load(inp)
        kmedoids = data['kmedoids']
        print("kmedoids bovw model opened")

    # add corresponding cluster label to each photo in df
    df['cluster'] = kmedoids.labels_

    # add True if to the rows that are medoids
    df['medoid'] = False
    df.loc[kmedoids.medoid_indices_, ['medoid']] = True

    return df


# show sse curve according to k
def elbow_plot(sse, elbow):
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), sse, label="SSE")
    plt.axvline(x=int(elbow), color='g', linestyle=':', linewidth=3)
    plt.title("Elbow")
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.legend()
    plt.show()


# Plot 2 photos of the same cluster c_num
def plot_clustered_photos(df, c_num):
    features = df.loc[:, ~df.columns.isin(['photo_name', 'blog_name', 'cluster'])]
    features = features.values.tolist()

    # sample 2 photos
    samples = df[df['cluster'] == c_num].sample(2)

    s1 = samples.iloc[0]
    s2 = samples.iloc[1]

    # read photo s1
    p1_path = s1['photo_name']
    p1 = io.imread(p1_path)

    # read photo s2
    p2_path = s2['photo_name']
    p2 = io.imread(p2_path)

    # plot sampled photos and their histograms
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(p1)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.subplot(222)
    plt.bar(range(df.shape[1] - 3), features[s1.name])

    plt.subplot(223)
    plt.imshow(p2)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.subplot(224)
    plt.bar(range(df.shape[1] - 3), features[s2.name])
    plt.show()
