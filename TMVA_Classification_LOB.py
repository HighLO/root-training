#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import getpass
import os
username = os.environ['USER']
if os.system('klist | grep Default | grep ' + username + '@CERN.CH'):
    os.system('echo %s' % getpass.getpass() + " | kinit " + username)


# In[ ]:


import numpy as np


# In[2]:


import ROOT


# ### Input data
# 
# Open input ROOT file and create a RDataFrame for filtering the data

# In[ ]:


#get_ipython().run_line_magic('jsroot', 'on')


# In[ ]:


#open remote file on the network using xrootd
file = ROOT.TFile.Open("root://eosproject//eos/project/h/highlo/workshop/ZS_ZC/messages/20150210.root")
#open local file 
#file = ROOT.TFile.Open("/data/moneta/highlo/20150210.root")


# In[ ]:


file.ls()


# In[ ]:


tree = file.Get("LOB")
tree.Print()


# Create `RDataFrame` from name of TTree and input file 

# In[ ]:


rdf = ROOT.RDataFrame("LOB", file)


# In[ ]:


rdfPrice = rdf.Filter('ZCN5_IsOriginalMessage == true && Price > 0').Define('Price_cent', 'Price ')


# ### Numpy Conversion
# 
# Get columns of data frame as Numpy arrays.
# The function `AsNumPy()` returns a dictionary of numpy arrays where the key is the column name

# In[ ]:


values = rdfPrice.AsNumpy(["Price_cent"])


# In[ ]:


print(values)


# In[ ]:


x = values["Price_cent"]
print(x,x.shape)


# ### Fill histograms with a Numpy array 
# 
# - Fill an histogram with distribution of array content (price data)
# - Fill an histogram as a sequence (time series)

# In[ ]:


h1 = ROOT.TH1D("h1","h1",100,1550,1650)


# In[ ]:


h1.FillN(10000,x.astype("float64"),np.ones(1000000),1)


# In[ ]:


h1.Draw()
ROOT.gPad.Draw()


# In[ ]:


h2 = ROOT.TH1D("h2","price vs time",10000,0,10000)
h2.FillN(10000,np.arange(0.,10000.),x[0:10000].astype('float64'))


# In[ ]:


#get_ipython().run_line_magic('jsroot', 'on')
h2.Draw('HIST')
ROOT.gPad.Draw()


# ### Data Manipulation
# 
# Using numpy manipulate the data to make a time series
# From the first N entry in the data, reshape in block of 10 values for each nb = N/10 row

# In[ ]:


nevt = 650000
seqLen = 13
nb = int(nevt/seqLen)
tmp = x[0:nevt].reshape(nb,seqLen)
print (tmp.shape)


# Split now in different arrays each colum of the tensor.
# The result is a list of 10 arrays XJ with the XJ(ievt) = X(ievt*10 + j) 

# In[ ]:


import numpy as np
r = np.hsplit(tmp,seqLen)


# Need to reshape each array to be of correct dimension

# In[ ]:


arrayList = []
for a in r: 
    aNew = a.reshape(nb)
    arrayList.append(aNew)
    print(aNew.shape,aNew)


# ### Create a new tree with new data sequences
# 
# Use `MakeNumpyDataFrame` to create a DataFrame and then a ROOT TTree with the new vectors
# 
# Use a Python dictionary to define the columns of the new tree 

# In[ ]:


nTime = 10
nTargets = 3
varDict = {}
for i in range(0,nTime):
    varName = 'x'+str(i)
    varDict[varName] = arrayList[0].astype('float32')
### add targets
for i in range(0,nTargets):
    varName = 'xtarget'+str(i)
    varDict[varName] = arrayList[nTime+i].astype('float32')


# In[ ]:


df = ROOT.RDF.MakeNumpyDataFrame(varDict)


# In[ ]:


df.Snapshot("tree","fileWithPriceSeq.root")


# ## Use TMVA
# 
# With TMVA use the data to classify if there was a price increase or decrease. 
# Make a binary classification splitting data in two classes depending if price increased or decreased

# In[3]:


file = ROOT.TFile.Open("fileWithPriceSeq.root")


# In[4]:


file.ls()


# In[5]:


tree = file.Get("tree")


# In[6]:


tree.Print()


# Examine the two types of data: 
#  1. Data where the price increase. All 3 futures prices are higher than last one
#  2.  Data where the proce decreases. All 3 futures proces are lower thna last one

# In[7]:


tree.Draw("x1 >> htmp1(100,1550,1650)","xtarget0 > x9 && xtarget1 > x9 && xtarget2 > x9")
ROOT.gPad.Draw()


# In[8]:


tree.Draw("x1 >> htmp2(100,1550,1650)","xtarget0 < x9  && xtarget1 < x9 && xtarget2 < x9" )
ROOT.gPad.Draw()


# #### Create the TMVA Factory
# 
# The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass
# 
# The first argument is the base of the name of all the output weightfiles in the directory weight/ that will be created with the method parameters
# 
# The second argument is the output file for the training results
# 
# The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string

# In[9]:


outputFile = ROOT.TFile("TMVA_classification.root","RECREATE")
factory = ROOT.TMVA.Factory( "TMVAClassification", outputFile,"!V:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" )


# #### Create the TMVA DataLoader
# 
# The DataLoader is the class to define and prepare the inputs for the ML Methods 
# 
# - Select input features using `AddVariable`
# - Define data labels for the classification: Signal and Background data.  
# - Provide splitting of training and test sample
# 

# In[10]:


dataloader = ROOT.TMVA.DataLoader("dataset")


# In[11]:


for i in range(0,10):
    varname = "x" + str(i)
    print("adding variable",varname)
    dataloader.AddVariable(varname, 'F' )


# In this example we will use a common tree and define the two classes using two separate filters.
# 
# We could use also seprate TTree and use `DataLoader::AddSignalTree` and `DataLoader::AddBackgroundTTree`

# In[12]:


cut1 = ROOT.TCut("xtarget0 > x9 && xtarget1 > x9 && xtarget2 > x9")
cut2 = ROOT.TCut("xtarget0 < x9 && xtarget1 < x9 && xtarget2 < x9")
dataloader.AddTree( tree,'Signal', 1.0, cut1 )
dataloader.AddTree( tree,'Background', 1.0, cut2 )


# In[13]:


cut = ROOT.TCut("")
dataloader.PrepareTrainingAndTestTree(cut, "nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:NormMode=NumEvents:!V" );


# ### Booking Methods
# 
# Here we book the TMVA methods. We will book
# - a Decision Tree method based n gradient boosting (BDTG)
# - a Recurrent neural network using an LSTM cell and built using TMVA DL
# - a similar network but built using Keras
# 
# We start Booking the Decison Tree. You can see all possible options in the TMVA Users Guide 

# In[14]:


factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:VarTransform=N:NTrees=1000::BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=3" )


# #### Make a TMVA recurrent network
# 
# We define here the Option string to configure the TMVA network based on a LSTM cell plus a dense layer for classification
# 
# ##### 1. Define network layout
# 
# The DNN configuration is defined using a string. Note that whitespaces between characters are not allowed. 
# 
# We define first the DNN layout: 
# 
# - **input layout** :   this defines the input data format. For a RNN  should be  ``sequence length | number of features``. 
#    
#    *(note the use of the character `|` as  separator of  input parameters for DNN layout)*
#                  
# - **layer layout** string defining the layer architecture. The syntax is  
#    - layer type (e.g. DENSE, CONV, RNN)
#    - layer parameters (e.g. number of units)
#    - activation function (e.g  TANH, RELU,...)
#    
#      *the different layers are separated by the ``","`` *
#                 
# #####  2. Define Training Strategy
# 
# We define here the training strategy parameters for the DNN. The parameters are separated by the ``","`` separator. 
# One can then concatenate different training strategy with different parameters. The training strategy are separated by 
# the ``"|"`` separator. 
# 
#  - Optimizer
#  - Learning rate
#  - Momentum (valid for SGD and RMSPROP)
#  - Regularization and Weight Decay 
#  - Dropout 
#  - Max number of epochs 
#  - Convergence steps. if the test error will not decrease after that value the training will stop
#  - Batch size (This value must be the same specified in the input layout)
#  - Test Repetitions (the interval when the test error will be computed) 
# 
# 
# ##### 3. Define general DNN options
# 
# We define the general DNN options concateneting in the final string the previously defined layout and training strategy.
# Note we use the ``":"`` separator to separate the different higher level options, as in the other TMVA methods. 
# In addition to input layout, batch layout and training strategy we add now: 
# 
# - Type of Loss function (e.g. cross entropy)
# - Weight Initizalization (e.g XAVIER, XAVIERUNIFORM, NORMAL )
# - Variable Transformation
# - Type of Architecture (e.g. CPU, GPU)
# 
# We can then book the method using the built otion string

# In[15]:


inputLayoutString = "InputLayout=10|1" # ntime, ninput
 
#Define RNN layer layout
#  it should be   LayerType (RNN or LSTM or GRU) |  number of units | number of inputs | time steps | remember output (typically no=0 | return full sequence
rnnLayout = "LSTM|12|1|10|0|1"
 
# add after RNN a reshape layer (needed top flatten the output) and a dense layer with 64 units and a last one
# Note the last layer is linear because  when using Crossentropy a Sigmoid is applied already
layoutString = "Layout=" + rnnLayout + ",RESHAPE|FLAT,DENSE|64|RELU,LINEAR";
 
# Defining Training strategies. Different training strings can be concatenate. Use however only one
training1 = ("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
            "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"
            "WeightDecay=1e-2,Regularization=None,MaxEpochs=50,"
            "Optimizer=ADAM,DropConfig=0.0+0.+0.+0.")


trainingStrategyString = "TrainingStrategy=" + training1

 
# Define the full RNN Noption string adding the final options for all network
# add a variable transformazion to normalize in interval [-1,1] input data
rnnOptions = ("!H:!V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                            "WeightInitialization=XAVIERUNIFORM:ValidationSize=0.2:RandomSeed=1234");

rnnOptions +=  ":" + inputLayoutString
#rnnOptions +=  ":" + batchLayoutString
rnnOptions +=  ":" + layoutString
rnnOptions +=  ":" + trainingStrategyString
rnnOptions +=  ":Architecture=GPU"
 
print(rnnOptions)


# In[16]:


factory.BookMethod(dataloader, ROOT.TMVA.Types.kDL, "TMVA_LSTM", rnnOptions)


# #### Make Keras model
# 
# We define now a similar model based on a LSTM using Keras

# In[17]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, SimpleRNN, GRU, LSTM, Reshape, BatchNormalization

model = Sequential() 
model.add(Reshape((10, 1), input_shape = (10, )))
model.add(LSTM(units=12, return_sequences=True) )
model.add(Flatten())
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(2, activation = 'sigmoid')) 
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
modelName = 'model_LSTM.h5'
model.save(modelName)
model.summary()


# In[18]:


ROOT.TMVA.PyMethodBase.PyInitialize()


# Book the PyKeras method using as input the Keras file defining the model : `model_LSTM.h5`

# In[19]:


factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras,"PyKeras_LSTM","!H:!V:VarTransform=N:FilenameModel=model_LSTM.h5:FilenameTrainedModel=trained_model_LSTM.h5:NumEpochs=30:BatchSize=100:GpuOptions=allow_growth=True")


# ### Train Methods
# 
# Here we train all the previously booked methods.

# In[ ]:


factory.TrainAllMethods()


# ### Test methods
# Here we test all methods using the test data set

# In[ ]:


factory.TestAllMethods()


# ### Evaluate methods
# 
# Here we evaluate all methods using both the test and training data set and we save results (histogram and output data tree's in the output file

# In[ ]:


factory.EvaluateAllMethods()


# # ###Plot result of Classification

# In[ ]:


cr = factory.GetROCCurve(dataloader)
cr.Draw()


# In[ ]:


outputFile.dataset.TestTree.Print()


# In[ ]:


outputFile.Close();
df = ROOT.RDataFrame('dataset/TestTree',outputFile.GetName())


# In[ ]:


r = df.Define("BDT_Output","BDTG")
h1 = r.Filter("classID==1").Histo1D(("h1","Output Class0",50,-1,1),'BDT_Output')
h2 = r.Filter("classID==0").Histo1D(("h2","Class1",50,-1,1),'BDT_Output')


# In[ ]:


h1.Draw()
h2.SetLineColor(ROOT.kRed)
h2.Draw("SAME")
ROOT.gPad.Draw()


# In[ ]:


r = df.Define("TMVA_Output","TMVA_LSTM")
h1 = r.Filter("classID==1").Histo1D(("h1","Output Class0",50,0,1),'TMVA_Output')
h2 = r.Filter("classID==0").Histo1D(("h2","Class1",50,0,1),'TMVA_Output')


# In[ ]:


h1.Draw()
h2.SetLineColor(ROOT.kRed)
h2.Draw("SAME")
ROOT.gPad.Draw()


# In[ ]:


r = df.Define("PyKeras_Output","PyKeras_LSTM")
h1 = r.Filter("classID==1").Histo1D(("h1","Output Class0",50,0,1),'PyKeras_Output')
h2 = r.Filter("classID==0").Histo1D(("h2","Class1",50,0,1),'PyKeras_Output')


# In[ ]:


h1.Draw()
h2.SetLineColor(ROOT.kRed)
h2.Draw("SAME")
ROOT.gPad.Draw()


# In[ ]:




