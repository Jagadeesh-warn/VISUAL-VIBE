import requests
import shutil
import os
import re
import pandas as pd
import pickle


# function that downloads an image from an url
def download_photos(image_url, image_name, dest_dir):
    # make a directory called images if not already exists
    os.makedirs(dest_dir, exist_ok=True)
    # get the extension of the image from url
    ext = re.search('[^.]+$', image_url).group()
    if ext == 'gif':
        ext = 'jpg'
    res = requests.get(image_url, stream=True)
    # check if the request was successfully
    if res.status_code == 200:
        # save the image as a binary file to the path "./images/"
        path = dest_dir + '/' + image_name + '.' + ext
        with open(path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image successfully Downloaded: ', image_name)
        return path
    else:
        print('Image Couldn\'t be retrieved')


# this class can be used to retrieve Tumblr data more easily
# using the OAuth2 API calls
class TumblrData:
    def __init__(self, oauth2=None):
        self.oauth2 = oauth2
        self.enuml = 0  # for enumerating liked photos
        self.enumc = 0  # for enumerating candidate photos

    # support function used to iterate over posts of type photo
    # photo_cat indicates whether photos are liked or not
    def _retrieve_photos(self, resp, photo_cat, name_prefix, folder, cluster=-1):
        local_paths = []  # local paths of downloaded liked photos
        blogs_name = []  # blogs name that posted the liked photos

        # iterate over posts
        for posts in resp[photo_cat]:
            # check if are of type photo
            if posts['type'] == 'photo':
                # if posts are already liked skip (valid only for candidate photos)
                if photo_cat == 'posts' and posts['liked'] is True:
                    continue
                # save the blog name of the current post
                b_name = posts['blog_name']
                print(b_name)
                # iterate over the current post photos
                for photos in posts['photos']:
                    # name current photo
                    curr_name = name_prefix
                    if photo_cat == 'posts':
                        self.enumc = self.enumc + 1
                        curr_name = curr_name + str(self.enumc)
                    else:
                        self.enuml = self.enuml + 1
                        curr_name = curr_name + str(self.enuml)
                    # save the url of the current photo
                    curr_url = photos['original_size']['url']
                    # download the liked photo
                    path = download_photos(curr_url, curr_name, folder)
                    # update the lists
                    local_paths.append(path)
                    blogs_name.append(b_name)

        # create a dataframe containing the data extracted earlier
        df = pd.DataFrame({'photo_name': local_paths, 'blog_name': blogs_name})

        return df

    # get tumblr photos liked by the user
    # photos are retrieved from last n liked posts
    def retrieve_liked_photos(self, n=20):
        if not os.path.exists('photo_liked'):
            likes = '/v2/user/likes'  # end point of liked photos
            self.enum = 0
            df = []
            for i in range(0, int(n / 20)):
                resp = self.oauth2.query(likes + '?offset=' + str(i * 20))['response']
                # specifies that are liked photos and where to store them
                df.append(self._retrieve_photos(resp, 'liked_posts', 'liked', 'photo_liked'))

            # save dataframe
            df_liked = pd.concat(df, ignore_index=True)
            with open("photo_liked/df_liked.pkl", 'wb') as out:
                pickle.dump({'liked': df_liked}, out)
            print("dataframe of liked photos saved")
            return df_liked
        else:
            with open("photo_liked/df_liked.pkl", 'rb') as inp:
                data = pickle.load(inp)
            print("dataframe of liked photos opened")
            return data['liked']

    # get last 'n' tumblr posts containing photos from blogs in blog_list
    # these photos will be matched with the ones already liked
    # cluster indicate which cluster these blogs belong to
    def retrieve_candidate_photos(self, blog_list, cluster, n=5):
        if not os.path.exists('photo_cand/cluster_' + str(cluster) + '/df_cand.pkl'):
            ep1 = '/v2/blog/'
            ep2 = '/posts/photo?limit=' + str(n)
            df = []
            # iterate over the blogs in the list
            for b in blog_list:
                resp = self.oauth2.query(ep1 + b + ep2)['response']
                # specifies that are just photos and where to store them
                # append the dataframe to the list named df
                df.append(self._retrieve_photos(resp, 'posts', 'cand', 'photo_cand/cluster_' + str(cluster), cluster))

            # save dataframe
            df_cand = pd.concat(df, ignore_index=True)
            with open('photo_cand/cluster_' + str(cluster) + '/df_cand.pkl', 'wb') as out:
                pickle.dump({'cand': df_cand}, out)
            print('dataframe of candidate photos for cluster ' + str(cluster) + ' saved')
            return df_cand
        else:
            with open('photo_cand/cluster_' + str(cluster) + '/df_cand.pkl', 'rb') as inp:
                data = pickle.load(inp)
            return data['cand']
