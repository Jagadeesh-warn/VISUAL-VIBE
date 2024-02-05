from tqdm import tqdm
#from cyvlfeat.sift import dsift
import cv2
import numpy as np
from skimage import io
from sklearn.cluster import MiniBatchKMeans
import warnings
import pickle
import os


# this class can be used to get a visual words representation
# of a given list of photos, hiding the steps of patch extraction
# description, clustering and photo patches occurrences computation
class BoVW:
    # extract all patches from photos in "training_photos" and compute their SIFT descriptor
    def _extract_and_describe(self):
        descriptors = []
        # for each photo append its SIFT descriptors to "descriptors"
        for i, row in tqdm(self.training_photos.iterrows(), 'Extracting/Describing Patches',
                           total=len(self.training_photos)):
            im = io.imread(row['photo_name'], as_gray=True)
            sift=cv2.SIFT_create()
            kp,desc=sift.detectAndCompute(im,None)
            #_, desc = dsift(im, size=self.size, step=self.step)
            descriptors.append(desc)
        # concatenate all the patch descriptors together
        return np.vstack(descriptors)

    # initialize the BoVW object and train the kmeans model with "training_photos"
    def __init__(self, training_photos, *, retrain=False, vocab_len=500, size=5, step=10):
        self.training_photos = training_photos
        self.vocab_len = vocab_len
        self.size = size
        self.step = step

        # if retrain is true then the training is performed
        # even if kmeans model can be found locally
        if retrain is False:
            try:
                with open("models/kmeans_desc.pkl", 'rb') as inp:
                    data = pickle.load(inp)
                self.kmeans = data['kmeans']
                print("kmeans dec model opened")
                return
            except OSError as e:
                if e.errno == 2:
                    print("kmeans desc model not found, retraining ...")
                else:
                    raise
        # extract descriptors
        all_training_desc = self._extract_and_describe()
        # suppress warnings
        warnings.filterwarnings('ignore')
        # initialize the optimized version of Kmeans called MiniBatchKMeans
        self.kmeans = MiniBatchKMeans(self.vocab_len, random_state=0)
        # fit the descriptors
        self.kmeans.fit(all_training_desc)

        #save the model
        os.makedirs('models', exist_ok=True)
        with open("models/kmeans_desc.pkl", 'wb') as out:
            pickle.dump({'kmeans': self.kmeans}, out)
        print("kmeans desc model saved")

    # from the trained model kmeans compute the tokens (cluster) of a given photo
    def _load_and_describe(self, img_name):
        im = io.imread(img_name, as_gray=True)
        sift=cv2.SIFT_create()
        kp,desc=sift.detectAndCompute(im,None)
        #_, descriptors = dsift(im, size=self.size, step=self.step)
        tokens = self.kmeans.predict(desc)
        return tokens

    # compute the bovw representation of photos in df
    def _get_representation(self, df, rep_name, cluster=-1):
        bovw_photos = []

        for i in range(0, len(df)):
            photo_i = df.iloc[i]['photo_name']
            tokens = self._load_and_describe(photo_i)
            # the representation is a list of occurrences of each token of the photo
            bovw_rep, _ = np.histogram(tokens, bins=self.vocab_len, range=(0, self.vocab_len - 1), density=True)
            bovw_photos.append(bovw_rep)

        # add the bovw representation to df as separated columns (are vocab_len in total)
        # each visual word has its own column
        for i in range(0, self.vocab_len):
            col_name = 'vw' + str(i + 1)
            df[col_name] = [p[i] for p in bovw_photos]

        # save dataframe of bovw representations
        os.makedirs('bovw_rep', exist_ok=True)
        if cluster != -1:
            os.makedirs('bovw_rep/cluster_'+str(cluster))
            with open('bovw_rep/cluster_'+str(cluster)+'/df_bovw_cand.pkl', 'wb') as out:
                pickle.dump({rep_name: df}, out)
        else:
            with open("bovw_rep/df_bovw_liked.pkl", 'wb') as out:
                pickle.dump({rep_name: df}, out)

        print("dataframe bovw saved")
        return df

    # compute bovw representation of liked photos
    # if not already done
    def get_liked_representation(self, df):
        if not os.path.exists('bovw_rep/df_bovw_liked.pkl'):
            return self._get_representation(df, 'liked')
        else:
            with open("bovw_rep/df_bovw_liked.pkl", 'rb') as inp:
                data = pickle.load(inp)
            return data['liked']

    # compute bovw representation of candidate photos
    # if not already done
    def get_candidate_representation(self, df, cluster):
        if not os.path.exists('bovw_rep/cluster_'+str(cluster)+'/df_bovw_cand.pkl'):
            return self._get_representation(df, 'cand', cluster)
        else:
            with open('bovw_rep/cluster_'+str(cluster)+'/df_bovw_cand.pkl', 'rb') as inp:
                data = pickle.load(inp)
            return data['cand']