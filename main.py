import tumblr_data
import bovw
import kmedoids
import cbir
consumer_key = '<API-KEY>'
consumer_secret = '<API-SECRET>'


if __name__ == '__main__':
    # LIKED PHOTOS EXTRACTION

    # get OAuth2 tokens (once photos are downloaded can be commented)
    # can be also removed as TumblrData param
    #obj_oauth2 = oauth2.Oauth2(consumer_key, consumer_secret)

    # download liked photos and return a dataset of local photo paths
    # along with blog name (photo author)
    # obj_oauth2 must be passed to TumblrData() if photos have to be downloaded
    tb = tumblr_data.TumblrData()
    pl = tb.retrieve_liked_photos(60)
    print(pl)

    # BAG OF VISUAL WORDS REPRESENTATION

    # create an object BovW
    obj_bovw = bovw.BoVW(pl)
    # get the bovw representation of liked photos
    pl = obj_bovw.get_liked_representation(pl)
    print(pl)

    # CLUSTERING WITH KMEDOIDS

    # add clusters label to pl
    pl = kmedoids.kmedoids_clustering(pl)
    print(pl)

    # plot two random photos from cluster 0
    kmedoids.plot_clustered_photos(pl, 0)


    # CBIR

    # get a dataset with:  query photo (with its cluster), best and worst recommendation
    rec = cbir.get_best_worst_recommendation(pl, tb, obj_bovw)
    print(rec)
    