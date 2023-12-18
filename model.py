import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression as LR
from tqdm import tqdm 
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statistics
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

def load_paris_data():
   
    data = pd.read_csv("paris_listings.csv")
   # print(data.info())
    to_drop = ['id', 'host_since', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description',
                'neighborhood_overview', 'picture_url', 'host_id', 'host_url', 'host_name',
                'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url', 
                'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
                'host_verifications', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                'latitude', 'longitude', 'bathrooms', 'maximum_nights', 'minimum_minimum_nights',
                'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
                'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped',
                'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 'license',
                'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
                'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
                'reviews_per_month', 'amenities', 'minimum_nights', 'number_of_reviews','host_acceptance_rate',
                'host_response_rate', 'review_scores_accuracy']
    
    data.drop(to_drop, inplace=True, axis=1)
    og_len = data.shape[0]
    #data['host_response_rate'] = data['host_response_rate'].str.rstrip("%").astype(float)/100

    #data['host_acceptance_rate'] = data['host_acceptance_rate'].str.rstrip("%").astype(float)/100
    data['review_scores_rating'] = data['review_scores_rating'].astype(float)/5
    data['review_scores_cleanliness'] = data['review_scores_cleanliness'].astype(float)/5
    data['review_scores_location'] = data['review_scores_location'].astype(float)/5
    data['review_scores_value'] = data['review_scores_value'].astype(float)/5
    data['review_scores_checkin'] = data['review_scores_checkin'].astype(float)/5
    data['review_scores_communication'] = data['review_scores_communication'].astype(float)/5

    data = data.replace({'host_is_superhost':{'t':1, 'f':0},
                          'host_identity_verified':{'t':1, 'f':0},
                          'host_has_profile_pic':{'t':1, 'f':0},
                          'instant_bookable':{'t':1, 'f':0}})

    data['price'] = data['price'].replace(
                                '[\$,]', '', regex=True).astype(float)
    data['bathrooms_text']= data['bathrooms_text'].str.findall(r'(\d+(?:\.\d+)?)')
    data['bathrooms_text']= data['bathrooms_text'].str[0].astype(float)
    

    data = data.fillna(value={'host_response_time': 'No replies'})
    #print(data['host_response_time'].value_counts())
    #print(data.isnull().sum())
    

    
    
    
    #print('Number of entries is {0:d} '.format(data.shape[0]))

    one_hot = pd.get_dummies(data[["property_type", "room_type", "host_response_time"]], drop_first = True).astype(int)
    data.drop(columns = ["property_type", "room_type", "host_response_time" ], inplace = True)
    data[one_hot.columns] = one_hot
    data.drop(columns = ['review_scores_rating', 'review_scores_location', 'review_scores_value', 'review_scores_checkin'], inplace=True)
    
    data = data.dropna()

   # print(data.head(5))
    #print(data.columns)
   #print('Orginial Number of entries: {:d}'.format(og_len))
    #print('Number of entries after data cleaing: {:d}'.format(data.shape[0]))
    #lost_entries = og_len - data.shape[0]
    #lost_entries_per = lost_entries/og_len*100
    #print('Number of entires lost: {:d} or {:.1f}%'.format(lost_entries, lost_entries_per)) 
    return data
def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """
    
    return np.sum(Y==Yhat)/len(Y)

def decisionTree(data):
        relevantFeatures = data.iloc[:,8:]
        y = data["price"]
        X_train, X_test, y_train, y_test = train_test_split(relevantFeatures, y, test_size=0.3, random_state=42)
        

       
        possible_cps = [0,0.00005,0.0001,0.0005,0.001]
        best_cp = 0
        best_accuracy_cp = 0
        for cp in possible_cps:
    
            accuracies = []
            for j in range(5):
                Xtrain_c, Xvalc, Ytrain_c, Yvalc = train_test_split(X_train, y_train, test_size=0.2, random_state=j)
                Xtrain_c.reset_index(inplace=True, drop=True)
                Xvalc.reset_index(inplace=True, drop=True)
                model = DTR(ccp_alpha=cp)
                model.fit(Xtrain_c, Ytrain_c)
                accuracies.append(accuracy(Yvalc, model.predict(Xvalc)))
    
            mean_accuracycp = sum(accuracies)/len(accuracies)
            if mean_accuracycp > best_accuracy_cp:
                best_accuracy_cp = mean_accuracycp
                best_cp = cp

        print("Best cp =", best_cp, "\n")

        possible_depths = [1, 2, 3, 4, 5]
        best_depth = 1
        best_accuracy = 0
        for depth in possible_depths:
    
            accuracies = []
            for i in range(5):
                Xtrain_i, Xval, Ytrain_i, Yval = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
                Xtrain_i.reset_index(inplace=True, drop=True)
                Xval.reset_index(inplace=True, drop=True)
                model = DTR(max_depth=depth, criterion='friedman_mse')
                model.fit(Xtrain_i, Ytrain_i)
                accuracies.append(accuracy(Yval, model.predict(Xval)))
    
            mean_accuracy = sum(accuracies)/len(accuracies)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_depth = depth

        print("Best depth =", best_depth, "\n")

        dtr = DTR(ccp_alpha=best_cp, criterion='friedman_mse',max_depth=best_depth )
        dtr.fit(X_train, y_train)
        
        ft_importance = pd.DataFrame({"feature": relevantFeatures.columns, "importance": dtr.feature_importances_}).sort_values("importance", ascending=False).query("importance > 0")
        print(ft_importance.shape[0])
        print(ft_importance.head(10))
        #dtr_score = dtr.score(X_train, y_train)
        dtr.predict(X_test)
        dtr_score = dtr.score(X_test,y_test)
        print(dtr_score)
        print(np.mean(y))
        #plt.figure(figsize=(8,8))
        ##ax = plot_tree(dtr, feature_names = relevantFeatures.columns, fontsize=20, max_depth=5)
        #plt.show()
        


        #dtr_score = dtrtest.score(X_test, y_test)
        

        
    

def main():
    load_paris_data()
    decisionTree(load_paris_data())
if __name__ == "__main__":
    main()
