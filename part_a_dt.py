# %%

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# %%
import sklearn
print(sklearn.__version__)

# %%
label_encoder = None 

# %%
def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()


# %%
X_train,y_train = get_np_array('./Dataset/train.csv')
X_test, y_test = get_np_array("./Dataset/test.csv")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]


# %%
class DTNode:

    def __init__(self, depth, is_leaf = False, value = 0, column = None, category = None, types = None):

        self.depth = depth
        self.children = None
        self.is_leaf = is_leaf
        self.value = value
        self.column = column
        
        self.category = category
        self.types = types
    
    def __call__(self,X):
        
        if self.is_leaf:
            return self.category

        if self.types[self.column] == 'cat':
            for child in self.children:
                if child.value == X[self.column]:
                    return child(X)
        else:
            if X[self.column] <= self.value:
                return self.children[0](X)
            else:
                return self.children[1](X)    
        
        


    def get_children(self, X, y, types, max_depth):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            None
        '''
        
        if self.is_leaf:
            majority = np.sum(y)/len(y)
            if majority > 0.5:
                self.category = 1
            else:
                self.category = 0
                
            return

        if np.sum(y) == len(y):
            self.is_leaf = True
            self.category = 1
            return 
        elif np.sum(y) == 0:
            self.is_leaf = True
            self.category = 0
            return
            
        
        max_gain = -1
        max_child_id = None
        
        ids = np.arange(len(types))
        ids = np.random.permutation(ids)
        
        for id in ids:
            type = types[id]
            
            if(type == 'cat'):
                #print("cat")
                categories = np.unique(X[:,id])
                current_gain = 0
                
                for category in categories:
                    category_indices = np.where(X[:,id] == category)
                    category_y = y[category_indices]
                    
                    positive_count = np.sum(category_y)
                    negative_count = len(category_y) - positive_count
                    total_count = len(category_y)
                    
                    this_category_gain = (positive_count/total_count)*np.log(positive_count/total_count) + (negative_count/total_count)*np.log(negative_count/total_count)
                    
                    current_gain += (total_count/len(y))*this_category_gain
                    
                
                if(current_gain > max_gain):
                    max_gain = current_gain
                    max_child_id = id
                
            else:
                current_gain = 0

                split_accross = np.median(X[:,id])
                split_accross_indices_more = np.where(X[:,id] <= split_accross)
                
                split_accross_indices_more_y = y[split_accross_indices_more]
                
                positive_count = np.sum(split_accross_indices_more_y)
                negative_count = len(split_accross_indices_more_y) - positive_count
                total_count = len(split_accross_indices_more_y)
                
                this_category_gain = (positive_count/total_count)*np.log(positive_count/total_count) + (negative_count/total_count)*np.log(negative_count/total_count)
                
                current_gain += (total_count/len(y))*this_category_gain
                
                
                split_accross_indices_less = np.where(X[:,id] > split_accross)
                split_accross_indices_less_y = y[split_accross_indices_less]
                
                positive_count = np.sum(split_accross_indices_less_y)
                negative_count = len(split_accross_indices_less_y) - positive_count
                total_count = len(split_accross_indices_less_y)
                
                this_category_gain = (positive_count/total_count)*np.log(positive_count/total_count) + (negative_count/total_count)*np.log(negative_count/total_count)
                
                current_gain += (total_count/len(y))*this_category_gain
                
                if(current_gain > max_gain):
                    max_gain = current_gain
                    max_child_id = id
        
        
        if types[max_child_id] == 'cat':
            categories = np.unique(X[:,max_child_id])
            children = []
            for category in categories:
                category_indices = np.where(X[:,max_child_id] == category)                
                child = DTNode(self.depth+1,
                               is_leaf = (self.depth >= max_depth),
                               value = category,
                               column=max_child_id,
                               types=types
                               )
                
                child.get_children(X[category_indices], y[category_indices], types, max_depth)

                    
                children.append(child)
            self.children = children
            

            
        else:
            split_accross = np.median(X[:,id])
            children = [] 
            child = DTNode(self.depth+1,
                           is_leaf = (self.depth >= max_depth),
                           value = split_accross,
                           column=max_child_id,
                           
                           )
            indices = np.where(X[:,id] <= split_accross)
            child.get_children(X[indices], y[indices], types, max_depth)

            child2 = DTNode(self.depth+1,
                           is_leaf = (self.depth >= max_depth),
                           value = split_accross,
                           column=max_child_id,
                            types=types
                           )
            indices = np.where(X[:,id] > split_accross)
            child2.get_children(X[indices], y[indices], types, max_depth)

            
            children.append(child)
            children.append(child2)
            self.children = children
        


# %%
class DTTree:

    def __init__(self):
        #Tree root should be DTNode
        self.root = DTNode(0)       

    def fit(self, X, y, types, max_depth = 10):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root.get_children(X, y, types, max_depth)

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y:  predicted classes
        '''
        return self.root(X)
    
    def post_prune(self, X_val, y_val):
        pass
        #TODO

# %%
max_depth = 10
tree = DTTree()
tree.fit(X_train,y_train,types, max_depth = max_depth)


