# %%
import cv2
import numpy as np
import glob
from cvxopt import matrix,solvers
from scipy.spatial.distance import cdist,pdist,squareform
import multiprocessing
import random
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

import matplotlib
matplotlib.style.use('ggplot')



# %% [markdown]
# # Part (a)

# %%
def preprocess(sample):
    sample =  (cv2.resize(sample, (16, 16), interpolation=cv2.INTER_CUBIC))/255.0
    return sample.flatten()

# %%
def read_sample(path):
    sample = cv2.imread(path)
    return preprocess(sample)

# %%
class0 = 2
class1 = 3

# %%
for iter, file in enumerate(glob.glob(f'./train/{class0}/*.jpg')):
    if iter == 0:
        print(file)
        X0 = read_sample(file)
    else:
        X0 = np.vstack((X0, read_sample(file)))

# %%
X0.shape

# %%
regulising_factor = 1

# %%
for iter, file in enumerate(glob.glob(f'./train/{class1}/*.jpg')):
    if iter == 0:
        X1 = read_sample(file)
        print(file)
    else:
        X1 = np.vstack((X1, read_sample(file)))

# %%
X1.shape

# %%
X0

# %%
-1*X0

# %%
only_X_data = np.vstack((X0, X1))

all_X = np.vstack((-1*X0, X1))
y = np.vstack((-1*np.ones((X0.shape[0], 1)), np.ones((X1.shape[0], 1))))

negative_test_cases = X0.shape[0]
positive_test_cases = X1.shape[0]
number_of_test_case = all_X.shape[0]

P = np.dot(all_X, all_X.T)
print(P.shape)

q = -1*np.ones((number_of_test_case, 1))
print(q.shape)
# q = q.T

# P = -1*P
# y

# %%
G = np.vstack((-1*np.eye(number_of_test_case), np.eye(number_of_test_case)))
h = np.vstack((np.zeros((number_of_test_case, 1)), regulising_factor*np.ones((number_of_test_case, 1))))
print(G.shape, h.shape)

# %%
A =  np.hstack((-1*np.ones((1, negative_test_cases)), np.ones((1, positive_test_cases))))
print(A.shape)
b = np.zeros((1))

# %%
P = matrix(P, tc='d')
q = matrix(q, tc='d')
G = matrix(G, tc='d')
h = matrix(h, tc='d')
A = matrix(A, tc='d')
b = matrix(b, tc='d')

# %%
sol = solvers.qp(P,q,G,h,A,b)

# %%
solution = np.array(sol['x'])

# %% [markdown]
# ## 1(a)

# %%
number_of_supp_vectors = np.sum(solution > 1e-3)
print(number_of_supp_vectors)

# %%
linear_supp_vec = [1 if i > 1e-3 else 0 for i in solution]

# %%
solution_with_non_zero_alphas = solution.copy()
solution_with_non_zero_alphas[solution_with_non_zero_alphas < 1e-6] = 0

# %%
the_alphas = y*solution_with_non_zero_alphas
weights = np.dot(the_alphas.T, only_X_data).T
b = np.mean(y - np.dot(only_X_data,weights))

# %%
print("Weights: ", weights)
print("Bias: ", b)  

# %%
correct = 0
total = 0

for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    # print(validate_on.shape)
    if np.dot(validate_on, weights) + b <= 0:
        correct += 1
        
for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    if np.dot(validate_on, weights) + b >= 0:
        correct += 1

# %% [markdown]
# ## 1(b)

# %%
print(f"Accuracy: {(100*correct)/total} %")

# %% [markdown]
# ## 1(c)

# %%
solution_vec = solution.T[0]
largest_alphas = np.argsort(solution_vec)[-6:]
x_is = only_X_data[largest_alphas]

for i, x in enumerate(x_is):
    img = x.reshape((16, 16, 3))

    img  = (img * 255).astype(np.uint8)
    filename = f'part_1_c_supports_{i}.png'
    cv2.imwrite(filename,img)
    

# %%
img = weights.T.reshape((16, 16, 3))
img = img/(sum(the_alphas)[0])

img  = (img * 255).astype(np.uint8)
filename = f'part_1_c_weight.png'
cv2.imwrite(filename,img)

# %% [markdown]
# ## 2(a)

# %%
gamma = 0.001

# %%
KERNEL_MATRIX = (np.outer(y,y))* (np.exp(-gamma * cdist(only_X_data, only_X_data, 'sqeuclidean')))

# v = pdist(only_X_data, 'sqeuclidean')
# KERNEL_MATRIX = np.exp(-1*gamma*squareform(v))

q = -1*np.ones((number_of_test_case, 1))
G = np.vstack((-1*np.eye(number_of_test_case), np.eye(number_of_test_case)))
h = np.vstack((np.zeros((number_of_test_case, 1)), regulising_factor*np.ones((number_of_test_case, 1))))
print(G.shape, h.shape)
A =  np.hstack((-1*np.ones((1, negative_test_cases)), np.ones((1, positive_test_cases))))
# print(A.shape)
b = np.zeros((1))


# %%
KERNEL_MATRIX = matrix(KERNEL_MATRIX, tc='d')
q = matrix(q, tc='d')
G = matrix(G, tc='d')
h = matrix(h, tc='d')
A = matrix(A, tc='d')
b = matrix(b, tc='d')


# %%
kernel_sol = solvers.qp(KERNEL_MATRIX,q,G,h,A,b)
# only_X_data

# %%
kernel_solution = np.array(kernel_sol['x'])

# %%
number_of_supp_vectors = np.sum(kernel_solution > 1e-3)
print(number_of_supp_vectors)

# %%
gauss_supp_vec = [1 if i > 1e-3 else 0 for i in kernel_solution]

# %%
kernel_solution

# %%
def infer_on_gaussian(X_to_infer_on, X_used_for_training,alpha_i_s,y,gamma = 0.001):
    KERNEL_MATRIX = (np.exp(-gamma * cdist(X_used_for_training, X_used_for_training, 'sqeuclidean')))
    bias = np.mean(y - np.dot(KERNEL_MATRIX,alpha_i_s*y))
    
    KERNEL_TO_INFER = (np.exp(-gamma * cdist(X_used_for_training, X_to_infer_on, 'sqeuclidean')))
    predictions = np.dot(KERNEL_TO_INFER.T, (alpha_i_s*y)) + bias

    # predictions = [1 if pred >=0 else -1 for pred in predictions]
    return predictions

# %%
correct = 0
total = 0
infer_actual = []

for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    if iter == 0:
        X_to_infer_on = validate_on
    else:
        X_to_infer_on = np.vstack((X_to_infer_on, validate_on))
    infer_actual.append(-1)
        
for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    X_to_infer_on = np.vstack((X_to_infer_on, validate_on))
    infer_actual.append(1)

predictions = infer_on_gaussian(X_to_infer_on, only_X_data,kernel_solution,y,gamma = 0.001)
predictions = [1 if pred >=0 else -1 for pred in predictions]


for pred, actual in zip(predictions, infer_actual):
    if pred == actual:
        correct += 1

print(f"Accuracy: {(100*correct)/total} %")

# %%
common_supp_vec_count = 0
for a,b in zip(linear_supp_vec,gauss_supp_vec):
    if a == 1 and b == 1:
        common_supp_vec_count += 1
    
print(f"Common Support Vectors: {common_supp_vec_count}. Linear had {sum(linear_supp_vec)} and Gaussian had {sum(gauss_supp_vec)} supports vectors.")

# %%
kernel_solution_vec = kernel_solution.T[0]
largest_alphas = np.argsort(kernel_solution_vec)[-6:]
x_is = only_X_data[largest_alphas]

for i, x in enumerate(x_is):
    img = x.reshape((16, 16, 3))

    img  = (img * 255).astype(np.uint8)
    filename = f'part_2_c_supports_{i}.png'
    cv2.imwrite(filename,img)
    

# %%
from sklearn import svm

ski_svm = svm.SVC(kernel='linear')
ski_svm.fit(np.vstack((X0, X1)), np.hstack((np.zeros((X0.shape[0])), np.ones((X1.shape[0])))))


# %%
support_vectors_count = ski_svm.support_vectors_


print(f"Number of support vectors: {support_vectors_count}")

# %%
support_indices = ski_svm.support_

common_supp_vec_count = 0

for index in support_indices:
    if solution_vec[index] > 1e-3:
        common_supp_vec_count += 1

print(f"No of common support vectors between linear : {common_supp_vec_count}")

# %%
ski_weights = ski_svm.coef_[0]
our_weights = weights.T[0]


diff = abs(ski_weights - our_weights)

img = diff.reshape((16, 16, 3))
# img = img
print(max(diff))
print(np.mean(abs(ski_weights)))

img  = (img * 255*100).astype(np.uint8)
filename = f'diff_weights.png'
cv2.imwrite(filename,img)

euclid_dist = np.linalg.norm(ski_weights[0] - our_weights)

print(euclid_dist)

bias_ski = np.mean(y - np.dot(only_X_data,ski_weights))
bias_us = np.mean(y - np.dot(only_X_data,our_weights))

print(bias_us, bias_ski, abs(bias_us - bias_ski))


# %%
correct = 0
total = 0

for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    # print(ski_svm.predict([validate_on]))
    if ski_svm.predict([validate_on]) == 0:
        correct += 1
        
for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    if ski_svm.predict([validate_on]) == 1:
        correct += 1


# %%
print(f"Accuracy with SKLearn: {(100*correct)/total} %")


# %%
ski_svm_gauss = svm.SVC(kernel='rbf', gamma=gamma)

ski_svm_gauss.fit(np.vstack((X0, X1)), np.hstack((np.zeros((X0.shape[0])), np.ones((X1.shape[0])))))

num_support_vectors = len(ski_svm_gauss.support_vectors_)
print(f"Number of Support Vectors: {num_support_vectors}")



# %%
support_indices = ski_svm_gauss.support_

common_supp_vec_count = 0

for index in support_indices:
    if kernel_solution_vec[index] > 1e-3:
        common_supp_vec_count += 1

print(f"No of common support vectors between gaussian : {common_supp_vec_count}")

# %%
correct = 0
total = 0

for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    # print(ski_svm.predict([validate_on]))
    if ski_svm_gauss.predict([validate_on]) == 0:
        correct += 1
        
for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
    total += 1 
    validate_on = read_sample(file)
    if ski_svm_gauss.predict([validate_on]) == 1:
        correct += 1


# %%
print(f"Accuracy with SKLearn on gaussian is: {(100*correct)/total} %")


# %% [markdown]
# # Multi Class classification

# %%
classification_dict = {}

# %%
def classify_between_two(class0,class1):
    print("Classifying between ", class0, " and ", class1)
    for iter, file in enumerate(glob.glob(f'./train/{class0}/*.jpg')):
        if iter == 0:
            # print(file)
            X0 = read_sample(file)
        else:
            X0 = np.vstack((X0, read_sample(file)))
        
    for iter, file in enumerate(glob.glob(f'./train/{class1}/*.jpg')):
        if iter == 0:
            X1 = read_sample(file)
            print(file)
        else:
            X1 = np.vstack((X1, read_sample(file)))
    
    only_X_data = np.vstack((X0, X1))

    all_X = np.vstack((-1*X0, X1))
    y = np.vstack((-1*np.ones((X0.shape[0], 1)), np.ones((X1.shape[0], 1))))

    negative_test_cases = X0.shape[0]
    positive_test_cases = X1.shape[0]
    number_of_test_case = all_X.shape[0]

    gamma = 0.001
    
    KERNEL_MATRIX = (np.outer(y,y))* (np.exp(-gamma * cdist(only_X_data, only_X_data, 'sqeuclidean')))
    
    q = -1*np.ones((number_of_test_case, 1))
    G = np.vstack((-1*np.eye(number_of_test_case), np.eye(number_of_test_case)))
    h = np.vstack((np.zeros((number_of_test_case, 1)), regulising_factor*np.ones((number_of_test_case, 1))))
    A =  np.hstack((-1*np.ones((1, negative_test_cases)), np.ones((1, positive_test_cases))))
    b = np.zeros((1))
    
    
    KERNEL_MATRIX = matrix(KERNEL_MATRIX, tc='d')
    q = matrix(q, tc='d')
    G = matrix(G, tc='d')
    h = matrix(h, tc='d')
    A = matrix(A, tc='d')
    b = matrix(b, tc='d')
    
    kernel_sol = solvers.qp(KERNEL_MATRIX,q,G,h,A,b)
    
    kernel_solution = np.array(kernel_sol['x'])
    
    for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
        validate_on = read_sample(file)
        if iter == 0:
            X_to_infer_on = validate_on
        else:
            X_to_infer_on = np.vstack((X_to_infer_on, validate_on))
        infer_actual.append(-1)
            
    for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
        validate_on = read_sample(file)
        X_to_infer_on = np.vstack((X_to_infer_on, validate_on))
        infer_actual.append(1)
    
    predictions = infer_on_gaussian(X_to_infer_on, only_X_data,kernel_solution,y,gamma = 0.001)
    
    prediciton_iterator = 0
    
    for iter, file in enumerate(glob.glob(f'./val/{class0}/*.jpg')):
        if file not in classification_dict:
            classification_dict[file] = []
        
        # print(predictions[prediciton_iterator])
        classify_as = class1 if predictions[prediciton_iterator] >=0 else class0
        classification_dict[file].append((classify_as,predictions[prediciton_iterator][0]))
        prediciton_iterator += 1
    
    for iter, file in enumerate(glob.glob(f'./val/{class1}/*.jpg')):
        if file not in classification_dict:
            classification_dict[file] = []
        
        classify_as = class1 if predictions[prediciton_iterator] >=0 else class0
        classification_dict[file].append((classify_as,predictions[prediciton_iterator][0]))
        prediciton_iterator += 1



# %%
# for i in range(0,3):
#     for j in range(0,3):
#         if i < j:
#             classify_between_two(i,j)

# classify_between_two(0,1)

# %% [markdown]
# ## Making it parallel

# %%
num_classes = 6

# %%
def classify_between_two_parallel(class_pair):
    
    class0,class1 = class_pair
    
    classification_dict = {}
    print("Classifying between ", class0, " and ", class1)
    for iter, file in enumerate(glob.glob(f'./train/{class0}/*.jpg')):
        if iter == 0:
            # print(file)
            X0 = read_sample(file)
        else:
            X0 = np.vstack((X0, read_sample(file)))
        
    for iter, file in enumerate(glob.glob(f'./train/{class1}/*.jpg')):
        if iter == 0:
            X1 = read_sample(file)
            print(file)
        else:
            X1 = np.vstack((X1, read_sample(file)))
    
    only_X_data = np.vstack((X0, X1))

    all_X = np.vstack((-1*X0, X1))
    y = np.vstack((-1*np.ones((X0.shape[0], 1)), np.ones((X1.shape[0], 1))))

    negative_test_cases = X0.shape[0]
    positive_test_cases = X1.shape[0]
    number_of_test_case = all_X.shape[0]
#     for j in range(0,3):
#         if i < j:
#             classify_between_two(i,j)X.shape[0]

    gamma = 0.001
    
    KERNEL_MATRIX = (np.outer(y,y))* (np.exp(-gamma * cdist(only_X_data, only_X_data, 'sqeuclidean')))
    
    q = -1*np.ones((number_of_test_case, 1))
    G = np.vstack((-1*np.eye(number_of_test_case), np.eye(number_of_test_case)))
    h = np.vstack((np.zeros((number_of_test_case, 1)), regulising_factor*np.ones((number_of_test_case, 1))))
    A =  np.hstack((-1*np.ones((1, negative_test_cases)), np.ones((1, positive_test_cases))))
    b = np.zeros((1))
    
    
    KERNEL_MATRIX = matrix(KERNEL_MATRIX, tc='d')
    q = matrix(q, tc='d')
    G = matrix(G, tc='d')
    h = matrix(h, tc='d')
    A = matrix(A, tc='d')
    b = matrix(b, tc='d')
    
    kernel_sol = solvers.qp(KERNEL_MATRIX,q,G,h,A,b)
    
    kernel_solution = np.array(kernel_sol['x'])
    
    X_to_infer_on  = None
    
    for i in range(num_classes):
        for iter, file in enumerate(glob.glob(f'./val/{i}/*.jpg')):
            validate_on = read_sample(file)
            if X_to_infer_on is None:
                X_to_infer_on = validate_on
            else:
                X_to_infer_on = np.vstack((X_to_infer_on, validate_on))
            
    
    predictions = infer_on_gaussian(X_to_infer_on, only_X_data,kernel_solution,y,gamma = 0.001)
    
    prediciton_iterator = 0
    
    for i in range(num_classes):
        for iter, file in enumerate(glob.glob(f'./val/{i}/*.jpg')):                
            classify_as = class1 if predictions[prediciton_iterator] >=0 else class0
            classification_dict[file] = classify_as,predictions[prediciton_iterator][0]
            prediciton_iterator += 1
    
    # result_dict[(class0, class1)] = classification_dict
    return classification_dict




# %%
# num_classes = 6

# class_pairs = [(i, j) for i in range(num_classes) for j in range(i+1, num_classes)]

# manager = multiprocessing.Manager()
# result_dict = manager.dict()

# pool = multiprocessing.Pool(processes=6)

# for class_pair in class_pairs:
#     pool.apply_async(classify_between_two_parallel, args=(class_pair[0], class_pair[1], result_dict))

# pool.close()
# pool.join()

# final_classification_dict = {}
# for classification_dict in result_dict.values():
#     for key, value in classification_dict.items():
#         if key not in final_classification_dict:
#             final_classification_dict[key] = []
#         final_classification_dict[key].extend(value)

# %%


class_pairs = [(i, j) for i in range(num_classes) for j in range(i+1, num_classes)]
print(class_pairs)
pool = multiprocessing.Pool(processes=15)

result_dicts = pool.map(classify_between_two_parallel, class_pairs)

pool.close()
pool.join()

# final_classification_dict = {}
# for classification_dict in result_dicts:
#     for key, value in classification_dict.items():
#         if key not in final_classification_dict:
#             final_classification_dict[key] = []
#         final_classification_dict[key].extend(value)

# %%
final_classification_dict = {}
for classification_dict in result_dicts:
    for key, value in classification_dict.items():
        if key not in final_classification_dict:
            final_classification_dict[key] = []
        current_array = final_classification_dict[key]
        final_classification_dict[key] = current_array + [value]

# %%
final_classification_dict

# %%

final_labels = {}
for file, classifications in final_classification_dict.items():
    
    votes = {}
    print(classifications)
    for classify_as, score in classifications:
        if classify_as not in votes:
            votes[classify_as] = (0,0)
        
        number_of_vote, score_till_now = votes[classify_as]
        votes[classify_as] = (number_of_vote + 1, score_till_now + abs(score))
        
    print(votes)
    max_votes = max(votes, key=lambda k: (votes[k][0], votes[k][1]))
    print(max_votes)
    # sys.exit(0)

    
    final_labels[file] = max_votes


# %%
# len(final_labels)
# final_labels
# final_labels['./val/0/21827.jpg']
final_labels

# %%
correct = 0
total = 0

mis_classify = []

for i in range(0,num_classes):
    for iter, file in enumerate(glob.glob(f'./val/{i}/*.jpg')):
        total += 1
        if final_labels[file] == i:
            correct += 1
        
        else:
            mis_classify.append((file,final_labels[file],i))

# %%
sampled_images = random.sample(mis_classify,12)

print('''\\begin{figure}[H]
    \centering''')

for iter,sample_img in enumerate(sampled_images):
    
    file_name, classified_as,actually_is = sample_img
    
    img = cv2.imread(filename=file_name)
    file_new = f'./Mis_classify/mis_class_{iter}.png'
    cv2.imwrite(file_new,img)
    
    
    print(f'''    \\begin{{subfigure}}{{0.25\\textwidth}}
        \includegraphics[width=\linewidth]{{mis_class_{iter}.png}}
        \caption{{Misclassied label {actually_is} as {classified_as} }}
    \end{{subfigure}}''')
    
    if iter % 4 == 3:
        print('     \\vspace{0.5cm}')
    else:
        print('     \hfill')
    
print('''    \caption{Misclassifications}
\end{figure}''')

# %%
total

# %%


# %%
print(f"Accuracy for multi classification is {(100*correct)/total}%")

# %%
X_multi_class = None
y = None

for i in range(0,num_classes):
    for iter, file in enumerate(glob.glob(f'./train/{i}/*.jpg')):
        if X_multi_class is None:
            X_multi_class = read_sample(file)
            y_multi_class = np.array([i])
        else:
            X_multi_class = np.vstack((X_multi_class, read_sample(file)))
            y_multi_class = np.hstack((y_multi_class, np.array([i])))

# %%
y_multi_class

# %%
ski_svm_gauss = svm.SVC(kernel='rbf', gamma=gamma, break_ties=True)

ski_svm_gauss.fit(X_multi_class,y_multi_class)

num_support_vectors = len(ski_svm_gauss.support_vectors_)
print(f"Number of Support Vectors: {num_support_vectors}")



# %%
correct = 0
total = 0

ski_predictions = []
ski_actual_labels = []

for i in range(0,num_classes):
    for iter, file in enumerate(glob.glob(f'./val/{i}/*.jpg')):
        total += 1 
        validate_on = read_sample(file)
        ski_predictions.append(ski_svm_gauss.predict([validate_on]))
        ski_actual_labels.append(i)
        if ski_predictions[-1] == i:
            correct += 1

# %%
print(f"Accuracy with SKLearn on gaussian for multi class is: {(100*correct)/total} %")


# %%
actual_labels = []
predicted_labels = []
for key, value in final_labels.items():
    _,_,x,_ = key.split('/')
    actual_labels.append(int(x))
    predicted_labels.append(value)
    

# %%
confusion = confusion_matrix(actual_labels, predicted_labels, labels=[i for i in range(6)])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', xticklabels=[i for i in range(6)], yticklabels=[i for i in range(6)])

plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')

plt.show()

# %%
confusion = confusion_matrix(ski_actual_labels, ski_predictions, labels=[i for i in range(6)])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', xticklabels=[i for i in range(6)], yticklabels=[i for i in range(6)])


plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')

plt.show()

# %%


# %% [markdown]
# ## 5-Fold cross validation

# %%
X_multi_class = None
y = None

for i in range(0,num_classes):
    for iter, file in enumerate(glob.glob(f'./train/{i}/*.jpg')):
        if X_multi_class is None:
            X_multi_class = read_sample(file)
            y_multi_class = np.array([i])
        else:
            X_multi_class = np.vstack((X_multi_class, read_sample(file)))
            y_multi_class = np.hstack((y_multi_class, np.array([i])))

# %%
train_data_len = len(X_multi_class)
indices = np.arange(train_data_len)

np.random.shuffle(indices)

chunk_indices = np.array_split(indices, 5)

X_chunks = [X_multi_class[chunk] for chunk in chunk_indices]
y_chunks = [y_multi_class[chunk] for chunk in chunk_indices]

# X_multi_class = X_multi_class[indices]
# y_multi_class = y_multi_class[indices]

# %%

# five_fold_cross_validation_accuracy = []
# validation_accuracy = []

# values_of_c = [1e-5 , 1e-3, 1, 5, 10]

# for c in values_of_c:

def cross_validation_on_c(c,gamma = 0.01):
    
    
    
    five_fold_cross_validation_accuracy = []
    validation_accuracy = []
    
    for i in range(5):
        
        print(f"Running K-Fold with {c}, holding out chunk {i} for validation")
        
        X_init = None
        
        for j in range(5):
            if j == i:
                continue
            if X_init is None:
                X_init = X_chunks[j]
                y_init = y_chunks[j]
            else:
                X_init = np.vstack((X_init, X_chunks[j]))
                y_init = np.hstack((y_init, y_chunks[j]))

        ski_svm_gauss = svm.SVC(kernel='rbf', gamma=gamma, C=c, break_ties=True)
        ski_svm_gauss.fit(X_init,y_init)
        
        predictions = ski_svm_gauss.predict(X_chunks[i])
        correct  = 0
        total  = 0
        for prediction, actual in zip(predictions, y_chunks[i]):
            if prediction == actual:
                correct += 1
            total += 1    
            
        five_fold_cross_validation_accuracy.append((100*correct)/total)
        
        correct = 0
        total = 0
        for i in range(0,num_classes):
            for iter, file in enumerate(glob.glob(f'./val/{i}/*.jpg')):
                total += 1 
                validate_on = read_sample(file)
                if ski_svm_gauss.predict([validate_on]) == i:
                    correct += 1
        
        validation_accuracy.append(((100*correct)/total))
    
    five_fold_cross_validation_accuracy_mean = np.mean(np.array(five_fold_cross_validation_accuracy))
    validation_accuracy_mean = np.mean(np.array(validation_accuracy))
    five_fold_cross_validation_accuracy_max = max(five_fold_cross_validation_accuracy)
    validation_accuracy_max = max(validation_accuracy)
    
    return (five_fold_cross_validation_accuracy_mean, validation_accuracy_mean,five_fold_cross_validation_accuracy_max,validation_accuracy_max)

# %%
# values_of_c = [1e-5 , 1e-3, 1, 5, 10]

values_of_c = [10**x for x in range(-5,5)]

with concurrent.futures.ThreadPoolExecutor(max_workers=len(values_of_c)) as executor:
    results_on_c = list(executor.map(cross_validation_on_c, values_of_c))


# %%
five_fold_cross_validation_accuracy = [v[0] for v in results_on_c]
validation_accuracy = [v[1] for v in results_on_c]
five_fold_cross_validation_accuracy_max = [v[2] for v in results_on_c]
validation_accuracy_max = [v[3] for v in results_on_c]

# %%
fig, ax = plt.subplots()
x_axis_at = [i for i in range(len(values_of_c))]

ax.plot(x_axis_at, five_fold_cross_validation_accuracy, label='5-Fold Accuracy', marker='o', linestyle='-')
ax.plot(x_axis_at, validation_accuracy, label='Validation accuracy', marker='s', linestyle=':')
ax.plot(x_axis_at, five_fold_cross_validation_accuracy_max, label='5-Fold accuracy max', marker='v', linestyle='-.')
ax.plot(x_axis_at, validation_accuracy_max, label='Validation accuracy max', marker='*', linestyle='--')

ax.set_xlabel('Value of C')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy on 5 fold, and validation ')

ax.set_xticks(x_axis_at)
ax.set_xticklabels([str(c) for c in values_of_c])

ax.legend()

plt.grid(True)
plt.show()


# %%
# from functools import partial

# def run_for_given_gamma(gamma):
#     # values_of_c = [1e-5 , 1e-3, 1, 5, 10]

#     cross_validation_on_c_for_gamma = partial(cross_validation_on_c, gamma = gamma)

#     values_of_c = [10**x for x in range(-5,5)]
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(values_of_c)) as executor:
#         results_on_c = list(executor.map(cross_validation_on_c_for_gamma, values_of_c))
        
#     five_fold_cross_validation_accuracy = [v[0] for v in results_on_c]
#     validation_accuracy = [v[1] for v in results_on_c]
#     five_fold_cross_validation_accuracy_max = [v[2] for v in results_on_c]
#     validation_accuracy_max = [v[3] for v in results_on_c]
    
#     fig, ax = plt.subplots()
#     x_axis_at = [i for i in range(len(values_of_c))]

#     ax.plot(x_axis_at, five_fold_cross_validation_accuracy, label='5-Fold Accuracy', marker='o', linestyle='-')
#     ax.plot(x_axis_at, validation_accuracy, label='Validation accuracy', marker='s', linestyle=':')
#     ax.plot(x_axis_at, five_fold_cross_validation_accuracy_max, label='5-Fold accuracy max', marker='v', linestyle='-.')
#     ax.plot(x_axis_at, validation_accuracy_max, label='Validation accuracy max', marker='*', linestyle='--')

#     ax.set_xlabel('Value of C')
#     ax.set_ylabel('Accuracy')
#     ax.set_title(f'Accuracy on 5 fold, and validation for gamma = {gamma} ')

#     ax.set_xticks(x_axis_at)
#     ax.set_xticklabels([str(c) for c in values_of_c])

#     ax.legend()

#     plt.grid(True)
#     plt.show()
    
#     return []



# %%
# values_of_gamma = [10**x for x in range(-5,5)]

# with concurrent.futures.ThreadPoolExecutor(max_workers=len(values_of_gamma)) as executor:
#     results_on_c = list(executor.map(run_for_given_gamma, values_of_gamma))


