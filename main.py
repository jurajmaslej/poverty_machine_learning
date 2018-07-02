import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import data_loader

from sklearn.metrics import confusion_matrix
import itertools
import copy

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



class Main:
	
	def __init__(self, data, header, validation_data):
		self.data = data[:,1:]
		print('new data shape ', self.data.shape)
		self.header = header
		self.validation_data = validation_data
		
	def normalize_data(self,data):
		#scaler = preprocessing.MinMaxScaler().fit(data)
		#scaler = preprocessing.StandardScaler().fit(data)
		#data = scaler.transform(data)
		data -=  np.mean(data, axis = 1, keepdims = True)
		data /= np.std(data, axis = 1, keepdims = True)
		return data
		
	def choose_categories(self, categories, target, for_validation=False):
		indexes = [self.header.index(i) for i in categories]
		target_index = self.header.index(target)
		
		categories_c = copy.deepcopy(categories)
		if target in categories_c:
			categories_c.remove(target)
		if for_validation:
			new_data = np.empty([len(self.validation_data), len(indexes)])
			i = 0
			for ind in indexes:
				new_data[:,i] = self.validation_data[:,ind]
				i+= 1
			self.data_pcaready_val = self.normalize_data(new_data)
			self.target_data_val = self.validation_data[:,target_index]
			return

		new_data = np.empty([len(self.data), len(indexes)])
		i = 0
		for ind in indexes:
			new_data[:,i] = self.data[:,ind]
			i+= 1
		
		self.data_pcaready = self.normalize_data(new_data)
		self.target_data = self.data[:, target_index]
		
	def classify_poverty(self,num_of_categories):
		#print(self.target_data.shape)
		#print(self.target_data_val.shape)
		all_data = np.hstack((self.target_data, self.target_data_val))
		minimum = min(all_data)
		maximum = max(all_data)
		difference = maximum - minimum
		step = difference / num_of_categories
		self.target_data_val = self.classify_poverty_into_categs(self.target_data_val, step)
		self.target_data = self.classify_poverty_into_categs(self.target_data, step)
		
	def classify_poverty_into_categs(self, data, step):
		classified = np.array([])
		for i in range(0,len(data)):
			category = int(data[i] / step)
			classified = np.append(classified, category)
		return classified
	
	def do_pca(self):
		#covariance, eigenvalues
		cor_mat1 = np.corrcoef(self.data_pcaready.T)

		eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

		#print('Eigenvectors \n%s' %eig_vecs)
		#print('\nEigenvalues \n%s' %eig_vals)
		
		# Make a list of (eigenvalue, eigenvector) tuples
		eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

		# Sort the (eigenvalue, eigenvector) tuples from high to low
		eig_pairs.sort()
		eig_pairs.reverse()
		sorted_eigvalues = []
		indexes = range(0,len(eig_pairs))
		# Visually confirm that the list is correctly sorted by decreasing eigenvalues
		#print('Eigenvalues in descending order:')
		for i in eig_pairs:
			sorted_eigvalues.append(i[0])
		plt.figure()
		plt.plot(indexes, sorted_eigvalues)
		plt.xlabel('category')
		plt.ylabel('eigenvalue')
		plt.savefig('eigenvalues_pca.png')
		
		#
		
		pca = decomposition.PCA(n_components=7)
		all_data = np.vstack((self.data_pcaready, self.data_pcaready_val))
		pca.fit(self.data_pcaready)
		self.pca_data = pca.transform(self.data_pcaready)
		#print(pca.get_covariance().shape)
		#print(pca.score(self.data_pcaready))
		#print(pca.components_.shape)
		pca.fit(self.data_pcaready_val)
		self.pca_data_val = pca.transform(self.data_pcaready_val)
		#print(self.pca_data)
		
	def do_lda(self):
		#at first scale data to only positive
		scaler = MinMaxScaler()
		
		lda = LatentDirichletAllocation(n_components=7)
		all_data = np.vstack((self.data_pcaready, self.data_pcaready_val))
		scaler.fit(all_data)
		all_data = scaler.transform(all_data)
		self.data_pcaready = scaler.transform(self.data_pcaready)
		self.data_pcaready_val = scaler.transform(self.data_pcaready_val)
		lda.fit(self.data_pcaready)
		self.pca_data = lda.transform(self.data_pcaready)
		
		lda.fit(self.data_pcaready_val)
		self.pca_data_val = lda.transform(self.data_pcaready_val)
		
	def do_svm_fit(self):
		clf = svm.SVC()
		clf.fit(self.pca_data, self.target_data)
		return clf
		
	def do_svm_predict(self, clf):
		predicted = clf.predict(self.pca_data_val)
		score = clf.score(self.pca_data_val, self.target_data_val)
		#print('score with pca ', score)
		return (predicted, score)
	
	def svm_kernel_testing(self):
		clf_list = []
		
		for fig_num, kernel in enumerate(('linear', 'rbf', 'sigmoid')):
			clf = svm.SVC(kernel=kernel, gamma = 10)
			clf.fit(self.pca_data, self.target_data)
			clf_list.append(clf)
			
		#gammas
		gammas = [2,10,30]
		cls_weights = dict()
		cls_weights[0] = 1
		cls_weights[1] = 1
		cls_weights[2] = 4
		cls_weights[3] = 3
		cls_weights[4] = 2
		
		for gm in gammas:
			clf = svm.SVC(kernel='rbf', gamma = gm, class_weight = cls_weights)
			clf.fit(self.pca_data, self.target_data)
			clf_list.append(clf)
		
		return clf_list
	
	def random_forest(self):
		n_estimators = [2,5,10,20,40]
		max_depth = [2,10,20,30,50]
		clf_list = []
		for est in n_estimators:
			for depth in max_depth:
				clf = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=0)
				clf.fit(self.pca_data, self.target_data)
				clf_list.append(clf)
		return clf_list
	
	def predict_for_list_of_models(self, models):
		best_score = 0
		best_p = None
		best_model = None
		for model in models:
			prediction, score = self.do_svm_predict(model)
			if score > best_score:
				best_score = score
				best_p = prediction
				best_model = model
		print('winning score ', str(best_score))
		#print('best model ' , best_model)
		#self.plot_predicted(best_p)
		return (best_p,model)
		
	def plot_predicted(self, predicted, fname):
		plt.figure()
		indexes = range(0,len(self.target_data_val[:50]))
		plt.plot(indexes,self.target_data_val[:50])
		plt.plot(indexes,predicted[:50])
		plt.savefig('first50' + fname + '.png')
		plt.close()
		
	def plot_confusion_matrix(self, prediction, filename):
		#print(prediction.shape)
		#print(self.target_data_val.shape)
		conf_matrix = confusion_matrix(self.target_data_val, prediction)
		cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
		#print("Normalized confusion matrix")
        
		cmap=plt.cm.Blues
		class_names= ['1', '2', '3', '4', '5']
		
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title('confusion_matrix ')
		#plt.colorbar()
		tick_marks = np.arange(len(class_names))
		plt.xticks(tick_marks, class_names, rotation=45)
		plt.yticks(tick_marks, class_names)

		fmt = '.2f'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig( filename + '_conf_matrix.png')
		plt.close()
		
	def plot_roc(self, classes_dict_pos, classes_dict_neg):
		for i in range(0,5):
			plt.figure()
			all_d = classes_dict_pos[i] + classes_dict_neg[i]
			#print ('all d ', all_d)
			plt.plot(['pos_true', 'pos_false', 'neg_true', 'neg_false'], all_d, 'ro')
			fname = 'classified_abs_numbers_class_' + str(i)
			plt.savefig(fname +'.png')
		
		
	def roc_curve(self, predicted):
		#	class:(true pos, false pos)
		classes_dict_pos = {0:(0,0),1:(0,0),2:(0,0),3:(0,0),4:(0,0)}
		for i in range(0, len(self.target_data_val)):
			tg = self.target_data_val[i]
			pr = predicted[i]
			if tg == pr:	#true positive
				classes_dict_pos[pr] = (classes_dict_pos[pr][0] + 1, classes_dict_pos[pr][1])
			else:
				classes_dict_pos[pr] = (classes_dict_pos[pr][0], classes_dict_pos[pr][1] + 1)
		#print (classes_dict_pos)
		
		classes_dict_neg = {0:(0,0),1:(0,0),2:(0,0),3:(0,0),4:(0,0)}
		for i in range(0, len(self.target_data_val)):
			tg = self.target_data_val[i]
			pr = predicted[i]
			if tg != pr:	#true neg
				classes_dict_neg[pr] = (classes_dict_neg[pr][0] + 1, classes_dict_neg[pr][1])
			else:
				classes_dict_neg[pr] = (classes_dict_neg[pr][0], classes_dict_neg[pr][1] + 1)
		#print (classes_dict_neg)
		return(classes_dict_pos, classes_dict_neg)
	
	
	def run_all(self):
		#print('svm_kernel_testing')
		list_of_svms = self.svm_kernel_testing()
		list_of_forests = self.random_forest()
		#print('predict_for_list_of_models')
		best_prediction_svm, best_svm = self.predict_for_list_of_models(list_of_svms)
		#print('plot conf matrix')
		self.plot_confusion_matrix(best_prediction_svm, 'svm')

		#print('predict_for_list_of_models')
		best_prediction_forest, best_forest = self.predict_for_list_of_models(list_of_forests)
		#print('plot conf matrix')
		self.plot_confusion_matrix(best_prediction_forest, 'rforest')
		classes_dict_pos_svm, classes_dict_neg_svm = self.roc_curve(best_prediction_svm)
		classes_dict_pos_fr, classes_dict_neg_fr =self.roc_curve(best_prediction_forest)

		self.plot_roc(classes_dict_pos_svm, classes_dict_neg_svm)


		self.plot_predicted(best_prediction_svm, 'svm')
		self.plot_predicted(best_prediction_forest, 'forest')
		
		
		
data_load = data_loader.Loader("consensus_county15.txt")
data = data_load.estimation_data
validation_data = data_load.validation_data[:,1:]
header = data_load.header[1:]
print(len(header))
pca = Main(data, header, validation_data)

#categories should stay the same
pca.choose_categories(header, 'Poverty')
pca.choose_categories(header, 'Poverty', for_validation=True)
#pca.choose_categories(['County', 'Income', 'White', 'TotalPop', 'Unemployment','PublicWork'], 'Poverty')

#pca.choose_categories(['County', 'Income', 'White', 'TotalPop', 'Unemployment','PublicWork'], 'Poverty', for_validation=True)


#print('classify_poverty')
pca.classify_poverty(5)

#print('do pca')
pca.do_pca()

#print('do lda')	# worse results
#pca.do_lda()
#for i in range(0,8):
#	pca.run_all()
pca.run_all()



