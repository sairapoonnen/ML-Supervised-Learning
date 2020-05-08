import numpy, matplotlib.pyplot, os.path
np = numpy
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback



def load_data(path='data'):
    train_data = numpy.genfromtxt(os.path.join(path,'adult.data'),delimiter=',',dtype=str)
    test_data = numpy.genfromtxt(os.path.join(path,'adult.test'),delimiter=',',dtype=str)
    return train_data[:,:-1], train_data[:,-1].reshape(-1,1), test_data[:,:-1], test_data[:,-1].reshape(-1,1)

def getNumMissing(data): 
    arr = data == " ?";
    arr = np.count_nonzero(arr == True ,axis= 1)
    return np.count_nonzero(arr != 0)

def removeMissing(dataX, dataY):
    arr = dataX == " ?";
    arr = np.count_nonzero(arr == True ,axis= 1)
    idx = np.where(arr == 0)[0]
    return (dataX[idx], dataY[idx])

def groupRegion(data):

    asiaEast = [" Cambodia", " China", " Hong", " Japan", " Laos", " Philippines", " Thailand", " Taiwan", " Vietnam"]
    middleEast = [" Iran"]
    asiaSouth = [" India"]
    latinAmerica = [" Columbia", " Cuba", " Ecuador", " Guatemala", " Jamaica", " Nicaragua", " Puerto-Rico",  " Dominican-Republic", " El-Salvador", " Haiti", " Honduras", " Mexico", " Peru", " Trinadad&Tobago"]  
    europe = [" Greece", " Italy", " Portugal", " Germany", " Poland", " Yugoslavia", " Hungary"," England", " Holand-Netherlands", " Ireland", " France", " Scotland"]
    us = [" Outlying-US(Guam-USVI-etc)", " United-States"]
    canada = [ " Canada"]
    misc = [" South"]

    tr = data[:,12]
    tr = np.where(np.isin(tr, asiaEast), "asiaEast", tr)
    tr = np.where(np.isin(tr, asiaSouth), "asiaSouth", tr)
    tr = np.where(np.isin(tr, middleEast), "middleEast", tr)
    tr = np.where(np.isin(tr, latinAmerica), "latinAmerica", tr)
    tr = np.where(np.isin(tr, europe), "europe", tr)
    tr = np.where(np.isin(tr, us), "us", tr)
    tr = np.where(np.isin(tr, canada), "canada", tr)
    tr = np.where(np.isin(tr, misc), "misc", tr)

    data[:,12] = tr
    return data


def oneHotEncode(data, data2): 

    
    extractedW = data[:,1]
    extractedW = np.reshape(extractedW, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedW)
    extractedW = enc.transform(extractedW).toarray()
    
    extractedB = data[:,2]
    extractedB = np.reshape(extractedB, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedB)
    extractedB = enc.transform(extractedB).toarray()

    extractedM = data[:,4]
    extractedM = np.reshape(extractedM, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedM)
    extractedM = enc.transform(extractedM).toarray()
 

    extractedO = data[:,5]
    extractedO = np.reshape(extractedO, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedO)
    extractedO = enc.transform(extractedO).toarray()

    extractedR = data[:,6]
    extractedR = np.reshape(extractedR, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedR) 
    extractedR = enc.transform(extractedR).toarray()


    extractedRa = data[:,7]
    extractedRa = np.reshape(extractedRa, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedRa)
    extractedRa = enc.transform(extractedRa).toarray()

    extractedS = data[:,8]
    extractedS = np.reshape(extractedS, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedS)
    extractedS = enc.transform(extractedS).toarray()

    extractedN = data[:,12]
    extractedN = np.reshape(extractedN, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(extractedN)
    extractedN = enc.transform(extractedN).toarray()

    edu_num = data[:,3]
    cap_gain = data[:,9]
    cap_loss = data[:,10]
    hours_week = data[:,11]

    retData = np.zeros((data.shape[0], 70))

    retData[:,0] = data[:,0]
    retData[:,1:8] = extractedW
    retData[:,8:24] = extractedB
    retData[:,24] = edu_num
    retData[:,25:32] = extractedM
    retData[:,32:46] = extractedO
    retData[:,46:52] = extractedR
    retData[:,52:57] = extractedRa
    retData[:,57:59] = extractedS
    retData[:,59] = cap_gain
    retData[:,60] = cap_loss
    retData[:,61] = hours_week
    retData[:,62:70] = extractedN
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data2)
    extractedL = enc.transform(data2).toarray()

    return (retData, extractedL[:,1])

def rf_tune(train_X, train_Y, test_X, test_Y):
   
   
    estimators = [50,75,100,200,400,600]
    max_features = [0.05,0.07,0.1,0.15,0.2]


    params = {'n_estimators': estimators, 'max_features': max_features}
    clf = GridSearchCV(RandomForestClassifier(), params, scoring = ['accuracy','f1'], refit = 'f1', cv =3, verbose = 1, n_jobs = -1)
    clf.fit(train_X, train_Y)
    # print(clf.best_params_)
    # print(clf.best_score_)
    # print(clf.cv_results_)
    scores_pd = pd.DataFrame(clf.cv_results_)
    scores_pd = scores_pd[['param_max_features', 'param_n_estimators','mean_fit_time','mean_score_time',  'std_test_accuracy', 'std_test_f1']]
    print(scores_pd)
    # print(clf.cv_results_['mean_test_accuracy'])
    # print(clf.cv_results_['mean_test_f1'])
    scores = clf.cv_results_['mean_test_accuracy'].reshape(len(max_features),len(estimators))
    # print(scores)

    scores2 = clf.cv_results_['mean_test_f1'].reshape(len(max_features),len(estimators))
    
    ax = sns.heatmap(scores, annot = True, fmt = '.5g')
    plt.xticks(np.arange(len(estimators)), estimators)
    plt.yticks(np.arange(len(max_features)), max_features)
    plt.xlabel('n_estimators')
    plt.ylabel('max_features')
    plt.title('Grid Search ACC Score')
    plt.show()

    ax = sns.heatmap(scores2, annot = True, fmt = '.5g')
    plt.xticks(np.arange(len(estimators)), estimators)
    plt.yticks(np.arange(len(max_features)), max_features)
    plt.xlabel('n_estimators')
    plt.ylabel('max_features')
    plt.title('Grid Search F1 Score')
    plt.show()


def svm_tune(train_X, train_Y, test_X,test_Y):


    # kernels = ['poly','rbf','sigmoid']
    c = [0.01,0.1,1,10,100]
    gamma = [0.0001,.01,0.1,1,10]
    params = {'C': c, 'gamma':gamma}
    clf = GridSearchCV(SVC(), params, scoring = ['accuracy','f1'], refit = 'f1', cv =3, verbose = 1, n_jobs = -1)
    clf.fit(train_X, train_Y)
    # print(clf.best_params_)
    # print(clf.cv_results_['mean_test_accuracy'])
    # print(clf.cv_results_['mean_test_f1'])

    scores = clf.cv_results_['mean_test_accuracy'].reshape(len(gamma),len(c))
    scores2 = clf.cv_results_['mean_test_f1'].reshape(len(gamma),len(c))

    scores_pd = pd.DataFrame(clf.cv_results_)
    scores_pd = scores_pd[['param_C', 'param_gamma','mean_fit_time','mean_score_time',  'std_test_accuracy', 'std_test_f1']]
    print(scores_pd)
    
    ax = sns.heatmap(scores, annot = True, fmt = '.4g')
    plt.yticks(np.arange(len(c)), c)
    plt.xticks(np.arange(len(gamma)), gamma)
    plt.ylabel('C')
    plt.xlabel('Gamma')
    plt.title('Grid Search ACC Score')
    plt.show()

    ax = sns.heatmap(scores2, annot = True, fmt = '.4g')
    plt.yticks(np.arange(len(c)), c)
    plt.xticks(np.arange(len(gamma)), gamma)
    plt.ylabel('C')
    plt.xlabel('Gamma')
    plt.title('Grid Search F1 Score')
    plt.show()

def plot_3D_proj(data, labels):
      
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection = '3d')
        pca = PCA(n_components = 3)
        proj_3d_data = pca.fit_transform(data)
        ax.scatter(proj_3d_data[:,0],proj_3d_data[:,1],proj_3d_data[:,2],c=labels)
        fig.gca().set_title('PCA 3D transformation')
        plt.show()


def nn_tune(train_X,train_Y, test_X, test_Y):


    lr = [0.0001,0.001,0.005,0.01, 0.05, 0.1]
    optimizersList = ['SGD','adam', 'RMSprop']
    params = {'optimizerS': optimizersList, 'learning_rate': lr}

    def bld_model(optimizerS, learning_rate):
        model = Sequential()
        model.add(Dense(activation='relu', input_dim=70, output_dim = 32))
        model.add(Dense(activation='relu', input_dim = 32, output_dim = 16))
        model.add(Dense(activation='relu', input_dim=16, output_dim = 4))
        model.add(Dense(activation='sigmoid', input_dim = 4, output_dim = 1))
        optimizer = None
        if optimizerS == 'adam':
            optimizer = optimizers.Adam(lr = learning_rate)
        elif optimizerS =='SGD':
            optimizer = optimizers.SGD(lr = learning_rate)
        else:
            optimizer = optimizers.RMSprop(lr = learning_rate)
        model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
        return model
    
    
    # model.fit(train_X, train_Y, nb_epoch = 20,  callbacks=[Metrics(validation=(train_X, train_Y))])
    # print(train_Y.shape)
    model = KerasClassifier(build_fn=bld_model,verbose=1)
    clf = GridSearchCV(estimator=model, param_grid=params,scoring = ['accuracy','f1'], refit = 'f1',n_jobs=-1, cv=3)
    clf.fit(train_X, train_Y, nb_epoch = 150)

    # print(clf.best_params_)

    scores = clf.cv_results_['mean_test_accuracy'].reshape(len(lr),len(optimizersList))
    scores2 = clf.cv_results_['mean_test_f1'].reshape(len(lr),len(optimizersList))

    scores_pd = pd.DataFrame(clf.cv_results_)
    scores_pd = scores_pd[['param_optimizerS', 'param_learning_rate','mean_fit_time','mean_score_time', 'std_test_accuracy', 'std_test_f1']]
    print(scores_pd)
    
    ax = sns.heatmap(scores, annot = True, fmt = '.4g')
    plt.xticks(np.arange(len(optimizersList)), optimizersList)
    plt.yticks(np.arange(len(lr)), lr)
    plt.ylabel('Learning Rate')
    plt.xlabel('Optimizer')
    plt.title('Grid Search ACC Score')
    plt.show()


    ax = sns.heatmap(scores2, annot = True, fmt = '.4g')
    plt.xticks(np.arange(len(optimizersList)), optimizersList)
    plt.yticks(np.arange(len(lr)), lr)
    plt.xlabel('Optimizer')
    plt.ylabel('Learning Rate')
    plt.title('Grid Search F1 Score')
    plt.show()
    

def test(train_X, train_Y, test_X, test_Y):

    print("Random Forest")
   
    rf = RandomForestClassifier(n_estimators = 200, max_features = 0.2)
    # rf.fit(train_X,train_Y)
    # predict = rf.predict(test_X)
    # f1 = f1_score(test_Y, predict)
    # accuracy_score_rf = accuracy_score(test_Y,predict)
    scores = cross_validate(rf, test_X, test_Y, cv = 30, scoring = ['f1', 'accuracy'])
    print("F1")
    f1 = scores['test_f1'].mean()
    rf_f1_std = scores['test_f1'].std()
    print(f1)
    print("Accuracy")
    accuracy_score_rf = scores['test_accuracy'].mean()
    rf_acc_std  = scores['test_accuracy'].std()
    print(accuracy_score_rf)
    interval_rf_f1 = 1.96 *  (rf_f1_std) / np.sqrt(30)
    interval_rf_acc = 1.96 *  (rf_acc_std) / np.sqrt(30)
    print("F1 confidence level")
    print(interval_rf_f1)
    print("Accuracy confidence level")
    print(interval_rf_acc)

    
    print("Support Vector Machine")
    sv = SVC(gamma= 0.01, C = 1)
    # sv.fit(train_X,train_Y)
    # predict = sv.predict(test_X)
    # f1_svm = f1_score(test_Y, predict)
    # accuracy_score_svm = accuracy_score(test_Y,predict)
    scores_sv = cross_validate(sv, test_X, test_Y, cv = 30, scoring = ['f1', 'accuracy'])
    print("F1")
    f1_svm = scores_sv['test_f1'].mean()
    svm_f1_std = scores_sv['test_f1'].std()
    print(f1_svm)
    print("Accuracy")
    accuracy_score_svm = scores_sv['test_accuracy'].mean()
    svm_acc_std  = scores_sv['test_accuracy'].std()
    print(accuracy_score_svm)
    interval_svm_f1 = 1.96 *  (svm_f1_std) / np.sqrt(30)
    interval_svm_acc = 1.96 *  (svm_acc_std) / np.sqrt(30)
    print("F1 confidence level")
    print(interval_svm_f1)
    print("Accuracy confidence level")
    print(interval_svm_acc)
  

    print("Neural Network")
    # model.fit(train_X, train_Y, nb_epoch = 100)
    # predict = model.predict(test_X)
    # labels = (predict > 0.5).astype(np.int)
    # f1_nn = f1_score(test_Y, labels)
    # accuracy_nn = accuracy_score(test_Y, labels)
    def bld_model():
        model = Sequential()
        model.add(Dense(activation='relu', input_dim=70, output_dim = 32))
        model.add(Dense(activation='relu', input_dim = 32, output_dim = 16))
        model.add(Dense(activation='relu', input_dim=16, output_dim = 4))
        model.add(Dense(activation='sigmoid', input_dim = 4, output_dim = 1))
        optimizer = optimizers.RMSprop(lr = 0.0001)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    model = KerasClassifier(build_fn=bld_model,epochs = 150, verbose=1)
    scores_nn = cross_validate(model, test_X, test_Y, cv = 30, scoring = ['f1', 'accuracy'])
    print("F1")
    f1_nn = scores_nn['test_f1'].mean()
    nn_f1_std = scores_nn['test_f1'].std()
    print(f1_nn)
    print("Accuracy")
    accuracy_nn = scores_nn['test_accuracy'].mean()
    nn_acc_std  = scores_nn['test_accuracy'].std()
    print(accuracy_nn)
    interval_nn_f1 = 1.96 *  (nn_f1_std) / np.sqrt(30)
    interval_nn_acc = 1.96 *  (nn_acc_std) / np.sqrt(30)
    print("F1 confidence level")
    print(interval_nn_f1)
    print("Accuracy confidence level")
    print(interval_nn_acc)
  



    df = pd.DataFrame([['F1','RF', f1],['F1','SVM',f1_svm],['F1','NN',f1_nn],['Accuracy','RF',accuracy_score_rf],
                   ['Accuracy','SVM', accuracy_score_svm],['Accuracy','NN',accuracy_nn]],columns=['Metric','Classifier','Score'])

    df.pivot("Classifier", "Metric", "Score").plot(kind='bar')
    plt.ylim(ymax = 0.95, ymin = 0.50)
    plt.show()

def timeGraphs(train_X, train_Y, test_X, test_Y):


    rf = RandomForestClassifier(n_estimators = 200, max_features = 0.2)
    startRF = np.datetime64('now')
    rf.fit(train_X,train_Y)
    endRF = np.datetime64('now')

    sv = SVC(gamma= 0.01, C = 1)
    startSV = np.datetime64('now')
    sv.fit(train_X,train_Y)
    endSV = np.datetime64('now')



    model = Sequential()
    model.add(Dense(activation='relu', input_dim=70, output_dim = 32))
    model.add(Dense(activation='relu', input_dim = 32, output_dim = 16))
    model.add(Dense(activation='relu', input_dim=16, output_dim = 4))
    model.add(Dense(activation='sigmoid', input_dim = 4, output_dim = 1))
    optimizer = optimizers.RMSprop(lr = 0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    startNN = np.datetime64('now')
    model.fit(train_X, train_Y, nb_epoch = 150)
    endNN = np.datetime64('now')

    durRF = (endRF - startRF).astype(int)
    durSVM = (endSV - startSV).astype(int)
    durNN = (endNN - startNN).astype(int)

    print("NN training")
    print(durNN)
    print("SVM training")
    print(durSVM)
    print("RF training")
    print(durRF)

    startRF = np.datetime64('now')
    predict = rf.predict(test_X)
    dur_rf_eval = (np.datetime64('now') - startRF).astype(int)

    startSV = np.datetime64('now')
    predict = sv.predict(test_X)
    dur_sv_eval = (np.datetime64('now') - startSV).astype(int)

    startNN = np.datetime64('now')
    predict = model.predict(test_X)
    labels = (predict > 0.5).astype(np.int)
    dur_nn_eval = (np.datetime64('now') - startNN).astype(int)

    print("SVM testing")
    print(dur_sv_eval)
    print("NN testing")
    print(dur_nn_eval)
    print("RF testing")
    print(dur_rf_eval)


    df = pd.DataFrame({"Classifier":["RF", "SVM", "NN"], "Time Elapsed": [durRF, durSVM, durNN]})
    df.plot.bar(x = "Classifier", y = "Time Elapsed")
    plt.title("Training Time")
    plt.show()

    df = pd.DataFrame({"Classifier":["RF", "SVM", "NN"], "Time Elapsed": [dur_rf_eval, dur_sv_eval, dur_nn_eval]})
    df.plot.bar(x = "Classifier", y = "Time Elapsed")
    plt.title("Testing Time")
    plt.show()



def main():

    train_X, train_Y, test_X, test_Y = load_data();
    


    ##See number of instances
    # print(train_X.shape)
    # print(test_X.shape)

    ## See number of missing instances
    # print(getNumMissing(train_X))
    # print(getNumMissing(test_X))  

    # STANDARD SCALER???
    ##Remove missing values and fnlwgt
    train_X, train_Y = removeMissing(train_X, train_Y)
    test_X, test_Y = removeMissing(test_X, test_Y)
    
    train_X = np.delete(train_X, 2, 1)
    test_X = np.delete(test_X, 2, 1)

    ##groupRegions
    train_X = groupRegion(train_X)
    test_X = groupRegion(test_X)

    ##Onehot encode
    train_X, train_Y = oneHotEncode(train_X, train_Y)
    test_X, test_Y = oneHotEncode(test_X,test_Y)
   
    # scaling
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # plot 3D transform of data
    # _,labels = numpy.unique(train_Y,return_inverse=True)
    # plot_3D_proj(train_X, labels)

    ##Random Forest Tuning
    # rf_tune(train_X, train_Y, test_X, test_Y)
    
    #SVM Tuning
    # svm_tune(train_X, train_Y,test_X, test_Y)

    ##Neural Network Tuning
    # nn_tune(train_X, train_Y, test_X, test_Y)

    # Run tests
    # test(train_X, train_Y, test_X, test_Y)

    # Graphs for training and evaluation time
    # timeGraphs(train_X,train_Y, test_X, test_Y)

if __name__ == '__main__':
    main()