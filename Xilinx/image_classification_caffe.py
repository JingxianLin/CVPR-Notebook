
# coding: utf-8

# 
# # Part 3: Putting it all together: Compile, Quantize and Deploy
# 
# In this Part of the lab, we will review what we learned in Parts 1 and 2, and deploy models to be accelerated on the FPGA. We will look at each step of the deployment process. 
# 
# Once you have the outputs of the compiler and quantizer, you will use the xfDNN deployment APIs to:
# 1. Open a handle for FPGA communication
# 2. Load weights, biases, and quantization parameters to the FPGA DDR
# 3. Allocate storage for FPGA inputs (such as images to process)
# 4. Allocate storage for FPGA outputs (the activation of the final layer run on the FPGA)
# 5. Execute the network
# 6. Run fully connected layers on the CPU
# 7. Run Softmax on CPU
# 8. Print the result (or send the result for further processing)
# 9. When you are done, close the handle to the FPGA
# 
# First, we will look at compiling, quantizing and deploying a Inception v1 image classification example. After completing the example, we will look at deploying a customer model, using the same steps. 
# 
# ### 1. Import required packages, check environment

# In[1]:


import os,sys,cv2
from __future__ import print_function

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Bring in Xilinx ML-Suite Compiler, Quantizer, PyXDNN
from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend as xfdnnCompiler
from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer
import xfdnn.rt.xdnn as pyxfdnn
import xfdnn.rt.xdnn_io as pyxfdnn_io

import warnings
warnings.simplefilter("ignore", UserWarning)

print("Current working directory: %s" % os.getcwd())
print("Running on host: %s" % os.uname()[1])
print("Running w/ LD_LIBRARY_PATH: %s" %  os.environ["LD_LIBRARY_PATH"])
print("Running w/ XILINX_OPENCL: %s" %  os.environ["XILINX_OPENCL"])
print("Running w/ XCLBIN_PATH: %s" %  os.environ["XCLBIN_PATH"])
print("Running w/ PYTHONPATH: %s" %  os.environ["PYTHONPATH"])
print("Running w/ SDACCEL_INI_PATH: %s" %  os.environ["SDACCEL_INI_PATH"])

get_ipython().system(u'whoami')
# Make sure there is no error in this cell
# The xfDNN runtime depends upon the above environment variables


# ### 2. Use a config dictionary to pass parameters
# 
# Similar to Parts 1 and 2, we will setup and use a config dictionary to simplify handing the arguments. In this cell, we will also perform some basic error checking. For this first example, we will attempt to classify a picture of a dog. 
# 

# In[3]:


config = {}

# Quick check to see if we are running on AWS, if not assume 1525 box
if os.path.exists("/sys/hypervisor/uuid"):
    with open("/sys/hypervisor/uuid") as fp:
        contents = fp.read()
        if "ec2" in contents:
            print("Runnning on Amazon AWS EC2")
            config["device"] = "aws"
else:
    print("Runnning on VCU1525")
    config["device"] = "1525"


config["images"] = ["../examples/classification/flower.jpg"] # Image of interest (Must provide as a list)

img = cv2.imread(config["images"][0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(config["images"])
plt.show()


# ### 3. Compile The Model
# 
# As in Part 1, we will use the xfDNN Compiler to compile the Inception v1 network defined in Caffe. Please refer to the parameter descriptions in Part 1 for guidance on these parameters. 
# 

# In[4]:


# Compiler Arguments

config["prototxt"] = "../models/caffe/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.prototxt" 
config["caffemodel"] = "../models/caffe/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel"
config["outmodel"] = "work/optimized_model" # String for naming optimized prototxt, caffemodel
config["fpgacommands"] = "work/fpga.cmds" # Compiler will generate FPGA instructions
config["memory"] = 5 # Available on-chip SRAM
config["dsp"] = 56 # Width of Systolic Array
config["ddr"] = 256 # Available off-chip DRAM

compiler = xfdnnCompiler(
    networkfile=config["prototxt"],       # Prototxt filename: input file
    weights=config["caffemodel"],         # Floating Point Weights: input file
    anew=config["outmodel"],              # String for intermediate prototxt/caffemodel
    generatefile=config["fpgacommands"],  # Script filename: output file
    memory=config["memory"],              # Available on chip SRAM within xclbin
    dsp=config["dsp"],                    # Rows in DSP systolic array within xclbin 
    ddr=config["ddr"]                     # Memory to allocate in FPGA DRAM for activation spill
)

# Invoke compiler
try:
    compiler.compile()

    # The compiler extracts the floating point weights from the .caffemodel. 
    # This weights dir will be stored in the work dir with the appendex '_data'. 
    # The compiler will name it after the caffemodel, and append _data
    config["datadir"] = "work/" + config["caffemodel"].split("/")[-1]+"_data"
        
    if os.path.exists(config["datadir"]) and os.path.exists(config["fpgacommands"]+".json"):
        print("Compiler successfully generated JSON and the data directory: %s" % config["datadir"])
    else:
        print("Compiler failed to generate the JSON or data directory: %s" % config["datadir"])
        raise
        
    print("**********\nCompilation Successful!\n")
    
    import json
    data = json.loads(open(config["fpgacommands"]+".json").read())
    print("Network Operations Count: %d"%data['ops'])
    print("DDR Transfers (bytes): %d"%data['moveops']) 
    
except Exception as e:
    print("Failed to complete compilation:",e)


# ### 4. Quantize The Model
# As in Part 2, we will use the xfDNN Quantizer to quantize the Inception v1 model defined in Caffe. Please refer to the parameter descriptions in Part 2 for guidance on these parameters. 
# 

# In[5]:


# Quantizer Arguments
#config["outmodel"] = Defined in Step 1 # String for naming intermediate prototxt, caffemodel
config["quantizecfg"] = "work/quantization_params.json" # Quantizer will generate quantization params
config["bitwidths"] = [16,16,16] # Supported quantization precision
config["in_shape"] = [3,224,224] # Images will be resized to this shape -> Needs to match prototxt
config["transpose"] = [2,0,1] # Transpose argument to quantizer
config["channel_swap"] = [2,1,0] # Channel Swap argument to quantizer
config["raw_scale"] = 255.0
config["img_mean"] = [104.007, 116.669, 122.679] # Mean of the training set (From Imagenet)
config["input_scale"] = 1.0
config["calibration_size"] = 15 # Number of calibration images quantizer will use
config["calibration_directory"] = "../xfdnn/tools/quantize/calibration_directory" # Directory of images

quantizer = xfdnnQuantizer(
    deploy_model=config["outmodel"]+".prototxt",          # Prototxt filename: input file
    weights=config["outmodel"]+".caffemodel",             # Floating Point weights
    output_json=config["quantizecfg"],                    # Quantization filename: output file
    bitwidths=config["bitwidths"],                        # Fixed Point precision: 8b or 16b
    dims=config["in_shape"],                              # Image dimensions [Nc,Nw,Nh]
    transpose=config["transpose"],                        # Transpose argument to caffe transformer
    channel_swap=config["channel_swap"],                  # Channel swap argument to caffe transfomer
    raw_scale=config["raw_scale"],                        # Raw scale argument to caffe transformer
    mean_value=config["img_mean"],                        # Image mean per channel to caffe transformer
    input_scale=config["input_scale"],                    # Input scale argument to caffe transformer
    calibration_size=config["calibration_size"],          # Number of calibration images to use
    calibration_directory=config["calibration_directory"] # Directory containing calbration images
)

# Invoke quantizer
try:
    quantizer.quantize()

    import json
    data = json.loads(open(config["quantizecfg"]).read())
    print("**********\nSuccessfully produced quantization JSON file for %d layers.\n"%len(data['network']))
except Exception as e:
    print("Failed to quantize:",e)


# ### 5. Deploy The Model
# Next, we need to utilize the xfDNN APIs to deploy our network to the FPGA. We will walk through the deployment APIs, step by step: 
# 1. Open a handle for FPGA communication
# 2. Load weights, biases, and quantization parameters to the FPGA DDR
# 3. Allocate storage for FPGA inputs (such as images to process)
# 4. Allocate storage for FPGA outputs (the activation of the final layer run on the FPGA)
# 5. Execute the network
# 6. Run fully connected layers on the CPU
# 7. Run Softmax on CPU
# 8. Print the result (or send the result for further processing)
# 9. When you are done, close the handle to the FPGA
# 
# First, we will create the handle to communicate with the FPGA and choose which FPGA overlay to run the inference on. For this lab, we will use the `xdnn_56_16b_5m` overlay. You can learn about other overlay options in the ML Suite Tutorials [here][].  
# 
# [here]: https://github.com/Xilinx/ml-suite

# In[6]:


# Create a handle with which to communicate to the FPGA
# The actual handle is managed by pyxfdnn

config["xclbin"] = "../overlaybins/" + config["device"] + "/xdnn_56_16b_5m.awsxclbin" # Chosen Hardware Overlay
## NOTE: If you change the xclbin, we likely need to change some arguments provided to the compiler
## Specifically, the DSP array width, and the memory arguments

config["xfdnn_library"] = "../xfdnn/rt/xdnn_cpp/lib/libxfdnn.so" # Library functions called by pyXFDNN

ret = pyxfdnn.createHandle(config['xclbin'], "kernelSxdnn_0", config['xfdnn_library'])
if ret:                                                             
    print("ERROR: Unable to create handle to FPGA")
else:
    print("INFO: Successfully created handle to FPGA")


# ### 6. Apply quantization scaling and transfer model weights to the FPGA. 

# In[7]:


# Quantize, and transfer the weights to FPGA DDR

# config["datadir"] = "work/" + config["caffemodel"].split("/")[-1]+"_data" # From Compiler
config["scaleA"] = 10000 # Global scaler for weights (Must be defined, although not used)
config["scaleB"] = 30 # Global scaler for bias (Must be defined, although not used)
config["PE"] = 0 # Run on Processing Element 0 - Different xclbins have a different number of Elements

(weightsBlob, fcWeight, fcBias ) = pyxfdnn_io.loadWeights(config)

# Note that this function returns pointers to weights corresponding to the layers that will be implemented in the CPU


# ### 7. Allocate space in host memory for inputs, load images from disk, and prepare images. 

# In[8]:


# Allocate space in host memory for inputs, Load images from disk

config["transform"] = "resize" # Resize Images to fit into network
config["firstfpgalayer"] = "conv1/7x7_s2" # Name of first layer to be ran on the FPGA -> Needs to match prototxt

(fpgaInputs, batch_sz) = pyxfdnn_io.prepareInput(config)


# ### 8. Allocate space in host memory for outputs

# In[9]:


# Allocate space in host memory for outputs

config["fpgaoutsz"] = 1024 # Number of elements in the activation of the last layer ran on the FPGA

fpgaOutputs = pyxfdnn_io.prepareOutput(config['fpgaoutsz'], batch_sz)


# ### 9. Write optimized micro-code to the xDNN Processing Engine on the FPGA. 

# In[10]:


# Write FPGA Instructions to FPGA and Execute the network!
if len(pyxfdnn._xdnnManager._handles) > 0: # Just make sure FPGA still available
    pyxfdnn.execute(
        config["fpgacommands"],
        weightsBlob,
        fpgaInputs,
        fpgaOutputs,
        batch_sz, # Number of images we are processing
        1, # Always leave this as 1
        config['quantizecfg'], 
        config['scaleB'], 
        config['PE']
    )


# ### 10. Execute the Fully Connected Layers on the CPU

# In[12]:


# Step 3.6
# Execute the Fully Connected Layers on the CPU
# The FPGA does not support fully connected layers
# Given they are very fast with BLAS in the CPU, we leave the final layers to be executed there

config["outsz"] = 102 # Number of elements output by FC layers
config["useblas"] = True # Accelerate Fully Connected Layers in the CPU

if len(pyxfdnn._xdnnManager._handles) > 0: # Just make sure FPGA still available
    fcOut = pyxfdnn.computeFC(
        fcWeight, 
        fcBias, 
        fpgaOutputs,
        batch_sz, 
        config['outsz'], 
        config['fpgaoutsz'], 
        config['useblas'] # Can use cblas if True or numpy if False
    )


# ### 11. Execute the Softmax layers

# In[13]:


# Compute the softmax to convert the output to a vector of probabilities
softmaxOut = pyxfdnn.computeSoftmax(fcOut, batch_sz)


# ### 12. Output the classification prediction scores

# In[14]:


# Print the classification given the labels synset_words.txt (Imagenet classes)

config["labels"] = "../models/caffe/flowers102/data/synset_words.txt"
pyxfdnn_io.printClassification(softmaxOut, config);

#Print Original Image for Reference 
img = cv2.imread(config["images"][0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(config["images"])
plt.show()


# ### 13. Close the handle 

# In[15]:


pyxfdnn.closeHandle()


# ### 14. Your Turn! 
# Great work! Now it is your turn! 
# 
# We have another trained model which leverages the Inception v1 architecture.    
# This one is trained on the flowers dataset which has 102 classes.  
# 
# The final, fully connected layer has only 102 outputs for 102 output categories.  
# 
# This means that the graph and weights are different.
# 
# Update this notebook to classify pretty flowers instead!
# 
# Start by clicking **Kernel** from the menu, and then select **Reset & Clear Output**. 
# 
# Next update the parameters in the following steps:   
# 
# ### In Step 2:
# Set `config["images"]` to a new image.  A test flower image is located here: `"../examples/classification/flower.jpg"`
# 
# ### In Step 3:
# Set `config["prototxt"]` to a new network: To classify flowers, use the prototxt located here: `"../models/caffe/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.prototxt"`   
# 
# Set `config["caffemodel"]` to a model trained to classify flowers. The flowers caffe model is located here: `"../models/caffe/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel"`
# 
# ### In Step 10:
# Set `config["outsz"]` to reflect the number of classification categories for flowers, which is `102`.  
# 
# ### In Step 12:
# Set `config["labels"]` with the flower labels. The labels are located here: `"../models/caffe/flowers102/data/synset_words.txt"`  
# 
# When you have made all the updates, click **Kernel** from the menu, and select **Restart & Run All** to see if you can classify the flower! 
