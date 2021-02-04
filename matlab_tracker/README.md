# TSBB17 Project 2: Visual Object Tracking
This repository contains skeleton code for the tracking project in TSBB17

## Getting started!

##### Clone the GIT repository:

   ```
   git clone https://gitlab.ida.liu.se/tsbb17/visual-object-tracking.git
   ``` 

##### Clone the submodules:  
   In the repository directory, run the commands:
   
   ```
   git submodule init  
   git submodule update  
   ```
  
   This will download the external libraries required. (Matconvnet[1] and Piotr's[2] toolbox)
   
##### Set up paths:

   Edit setup_env.m to set the correct paths for your machine, pointing out the 
   path to the dataset.
   
##### Compile the libraries   

   Run setup_env.m to compile the external libraries. This will create a file named 'installed' in external folder to indicate that the libraries are compiled. 
   If you want to recompile, **delete** this file before running setup_env.m again.

##### Run!
   Run demo.m. If the installation is correct, you should see NCC tracker running on the Soccer sequence.
   
## Implementing your own tracker
   The toolkit contains an implementation of the NCC tracker. You can use the NCC as a base, copying the files to a new folder, for example "MyAwsomeTracker". The folder name should be the name of the tracker.
   
   The directory should be organized as follows:  
   .   
   trackers/    
   |\_\_\_\_NCC/    
   |\_\_\_\_MyAwesomeTracker/  
   |- - - - |\_\_\_\_code/                                        
   |- - - - - - - - -|\_\_\_\_initialize.m                       
   |- - - - - - - - -|\_\_\_\_trackFrame.m                       
   |- - - - |\_\_\_\_parameters/                                
   |- - - - - - - - -|\_\_\_\_setting1.m  
   |- - - - - - - - -|\_\_\_\_setting2.m   
     
     
      
      
       
initialize.m is called only in the first frame and should do the initial setup   
trackFrame.m is called in every frame and should do the actual tracking  
parameters folder can contain different parameter settings for the tracker  


## Dataset
The toolkit can run the trackers on otb_mini dataset, which is a subset of the OTB dataset[3] and contains 30 sequences. The script util/load_all_sequences.m contains the information 
about all the sequences, while the anno/ folder contains the groundtruths for all the sequences.  
If you want to add your own sequence, modify util/load_all_sequences.m and add the groundtruth in the anno folder.


**NOTE** The dataset is available at the ISY lab computers ("/site/edu/bb/TSBB17/otb_mini").  

## Evaluating your tracker
You can modify the demo.m file to run your tracker. The script will run the specified trackers on the specified sequences. The script will also plot the success plot[3], returns the area under the plot (AUC) score, 
which you can use to rank the trackers. The script also returns mean overlap scores for each sequence and each tracker, which you can use for detailed analysis.  

## Feature extraction
The toolkit also contains a general feature extraction framework. Currently, four types of features are included:
1. Colorspace features. Currently only grayscale features.
2. HOG[4] features. It uses the Piotr's toolbox [2].
3. Lookup table features. These are implemented as a lookup table that directly maps an RGB to a feature vector. Lookup table for Color Names [5] is included.
4. Deep features, see separate subsection below.

You can easily incorporate your own features in the toolkit by adding a corresponding "get_featureX.m" function. For example, you can incorporate deep features by adapting the "get_featureX.m" function.   



Please check feature_extraction/extract_features.m for detailed information about how to use the feature extractor.  

###### Deep Feature Extraction
The script "feature_extraction/deep_features_demo.m" provides an example of extracting features from a deep network using MatConvNet (a MATLAB based deep learning library). Refer to this script when implementing your own feature extraction code for deep features.

###### Supplied Deep Networks
We provide four networks that are modified where needed to be useful as feature extractors. Architectures such as AlexNet, and VGG have fully connected layers which fixes the input image size to be 224x224. Further some layers in these networks do not use padding. As a result some of the border regions of the image are ignored. To avoid these issues, we have modified AlexNet and VGG-M by  i) Adding appropriate padding to all layers and ii) Removing fully connected layers. These modified networks can be used as feature extractors on images of any sizes. 

The modified networks for AlexNet and VGG-M are stored at  **("/site/edu/bb/TSBB17/networks")** along with standard GoogLeNet and ResNet.

More pre-trained networks can be obtained at: 
http://www.vlfeat.org/matconvnet/pretrained/


## References  
[1] Webpage: http://www.vlfeat.org/matconvnet/  
    GitHub: https://github.com/vlfeat/matconvnet  
    
    
[2] Piotr Dollár.  
    "Piotr’s Image and Video Matlab Toolbox (PMT)."  
    Webpage: https://pdollar.github.io/toolbox/  
    GitHub: https://github.com/pdollar/toolbox    
    
[3] Y. Wu, J. Lim, and M.-H. Yang.  
    Online object tracking: A benchmark.  
    In CVPR, 2013.  
    https://sites.google.com/site/trackerbenchmark/benchmarks/v10   
   
[4] N. Dalal, and B. Triggs.  
    Histograms of oriented gradients for human detection.   
    In CVPR 2005.  
    
[5] J. van de Weijer, C. Schmid, J. J. Verbeek, and D. Larlus.  
    Learning color names for real-world applications.  
    TIP, 18(7):1512–1524, 2009.      
    

    
