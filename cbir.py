import pandas as pd
from sklearn.neighbors import KDTree
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


# extract only visual words features from df row in range (start, end)
def extract_vw_features(df):
    return df.loc[:, ~df.columns.isin(['photo_name', 'blog_name', 'cluster', 'medoid'])]


# function that aggregates blogs name in a list
# without duplicates
def aggregate_blogs(df):
    return list(set(df['blog_name']))


# get candidates photos for each cluster
# (retrieved from blogs in the cluster)
def prepare_candidates(pl, tb, obj_bovw):
    # get blogs name for each cluster
    blogs_name = pl.groupby(['cluster']).apply(aggregate_blogs)
    # get number of clusters
    k = len(set(pl['cluster']))
    candidates = []
    # make candidates list
    for i in range(k):
        # after retrieving candidates represent them with bovw
        curr_cand = tb.retrieve_candidate_photos(blogs_name[i], i)
        curr_cand_bovw = obj_bovw.get_candidate_representation(curr_cand, i)
        candidates.append(curr_cand_bovw)
    return candidates


# get the medoid for each cluster
def prepare_queries(pl):
    # get number of clusters
    k = len(set(pl['cluster']))
    queries = []
    # make queries medoids list
    for i in range(k):
        q = pl.query('medoid==1 and cluster==' + str(i))
        queries.append(q)
    return queries


def get_best_worst_recommendation(pl, tb, obj_bovw):
    queries = prepare_queries(pl)
    candidates = prepare_candidates(pl, tb, obj_bovw)
    n_clust = len(queries)

    photo_name = []  # queries photo name
    best_match = []  # best matches photo name
    worst_match = []  # worst matches photo name
    cluster = []  # queries cluster
    for i in range(n_clust):
        # extract vw features from query and candidates
        cand_feat = extract_vw_features(candidates[i])
        query_feat = np.array(extract_vw_features(queries[i]))

        # create a tree that memorizes and indexes all candidates representations
        tree = KDTree(cand_feat)

        # compute distance and index of all candidates
        distance, index = tree.query(query_feat, k=len(candidates[i]))

        # get the index of the closest photo to query
        index_closest = index[0][0]

        # get the index of the farthest photo to query
        index_farthest = index[0][len(candidates[i]) - 1]

        # append current query, cluster, best and worst match to the lists
        photo_name.append(queries[i]['photo_name'].values[0])
        best_match.append(candidates[i].iloc[index_closest]['photo_name'])
        worst_match.append(candidates[i].iloc[index_farthest]['photo_name'])
        cluster.append(queries[i]['cluster'].values[0])

    # create a dataframe with best and worst match for each query
    df = pd.DataFrame(
        {'query': photo_name, 'best_match': best_match, 'worst_match': worst_match,
         'cluster': cluster})

    # plot queries along with best and worst matches
    plot_best_worst_recommendations(df)

    return df


# plot best and worst match given a query
def plot_best_worst_recommendations(df):
    # plot results
    for i in range(len(df)):
        plt.figure(figsize=(16, 8))

        # plot queries
        query_im = io.imread(df.iloc[i]['query'])
        plt.subplot(1, 3, 1)
        plt.imshow(query_im)
        plt.axis('off')
        plt.title('Query ' + str(i + 1))

        # plot best matches
        best_match_im = io.imread(df.iloc[i]['best_match'])
        plt.subplot(1, 3, 2)
        plt.imshow(best_match_im)
        plt.axis('off')
        plt.title('Best match ' + str(i + 1))

        # plot worst matches
        best_match_im = io.imread(df.iloc[i]['worst_match'])
        plt.subplot(1, 3, 3)
        plt.imshow(best_match_im)
        plt.axis('off')
        plt.title('Worst match ' + str(i + 1))

        plt.show()
