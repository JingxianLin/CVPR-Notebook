
# coding: utf-8

# # Part 1: Using the xfDNN Compiler
# 
# ## Introduction
# In this part, you will learn what steps are required to prepare and compile a network and model. Before being able to deploy networks/models to Xilinx FPGAs you will need to compile them.  
# 
# This step is more than just converting the framework graph representation to one Xilinx can execute. The xfDNN Compiler is a high performance optimizer for Machine Learning inference. Some of the techniques it performs are fusing and merging layers, optimizing memory usage and pre-scheduling complete network deployment. These techniques increase inference rates and lower inference latency. 
# 
# Using the xfDNN Compiler is an offline process, which only needs to be performed once per network. As you will see, the process is simple and quick. 
# 
# First, we will look at an example already ready to be run based on Inception v1. Lets look at each step: 
# 
# ### 1. Import the required packages  

# In[1]:


import os,sys
from __future__ import print_function

# Bring in Xilinx ML-Suite Compiler
from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend as xfdnnCompiler


# 
# ### 2. Define a new xfdnnCompiler instance and pass arguments  
# To simplify handling of arguments, a config dictionary is used. Take a look at the dictionary below. 
# 
# The arguments that need to be passed are: 
# - `prototxt` - Caffe representation of the network
# - `caffemodel` - Pre-trained Model for the network 
# - `outmodel` - Filename to save the prototxt of the optimized network
# - `fpgacommands` - Filename to save micro-instruction produced by the compiler needed to deploy
# - `memory` - Parameter to set the on-chip memory for the target xDNN overlay. This example will target an overlay with 5 MB of cache. 
# - `dsp` - Parameter to set the size of the target xDNN overlay. This example uses an overlay of size 56x32 DSPs. 
# - `ddr` - Amount of off-chip memory available. This example will allow the compiler to use up to 256 MB on DDR. 
# 
# The xfDNN Compiler interfaces with Caffe to read a network graph, and generates a sequence of instructions for the xfDNN Deploy APIs to execute on the FPGA.  
# 
# During this process the xfDNN Compiler performs computational graph traversal, node merging and optimization, memory allocation and optimization and, finally, micro-instruction generation.
#   

# In[2]:


# Use a config dictionary to pass parameters to the compiler
config = {}

# Compiler Arguments
config["prototxt"] = "../models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt" 
config["caffemodel"] = "../models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel"
config["outmodel"] = "work/optimized_model" # String for naming intermediate prototxt, caffemodel
config["fpgacommands"] = "work/bvlc_googlenet_without_lrn.cmd" # Compiler will generate FPGA instructions
config["memory"] = 5 # Available on-chip SRAM
config["dsp"] = 56 # Width of Systolic Array
config["ddr"] = 256 # Available off-chip DRAM

compiler = xfdnnCompiler(
    verbose=False,
    networkfile=config["prototxt"],       # Prototxt filename: input file
    weights=config["caffemodel"],         # Floating Point Weights: input file
    anew=config["outmodel"],              # Filename for optimized prototxt/caffemodel
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


# ### 3. Try it yourself with a different model
# 
# Now that you have had a chance to see how this works, it's time to get some hands on experience.
# Change the following from the example above: 
#     1. The Network - From Inception v1 to ResNet50. 
#     2. The Weights - New Network, will require to us to re-extract the weights. 
#     3. The amount of on-chip memory available. 
#     4. The size of the xDNN overlay. 
# 
# Resources: 
# ResNet50 prototxt located here: `"../models/caffe/resnet/fp32/resnet50_deploy.prototxt"`  
# ResNet50 caffemodel located  here:  `"../models/caffe/resnet/fp32/resnet50.caffemodel"`  
# 
# In the last example, you ran through the compiler using a setting of 5 MB. Try `3` for 3 MB of on-chip memory. This will restrict the memory available to the compiler and it will automatically create DDR transfer commands to move intermediate results between the FPGA and DDR. You will see the DDR transfers is no longer 0 bytes.
# 
# Lastly, 56 or 28 are supported as the DSP argument, so change this to 28. This corresponds to the 28x32 DSP configuration which reduces the number of DSPs compared to 56x32, but allows us to run twice as many CNN processing engines.
# 
# Note: Because Resnet-50 has more opportunities to optimize the graph, and the model parameters are approximately 2x the size of the GoogLeVet-v1 model, it will take slightly longer to write the optimized weights run compared to the previous GoogLeNet-v1 example.

# In[3]:


# Since we already have an instance of the compiler, you can just update these params:

compiler.networkfile = "../models/caffe/resnet/fp32/resnet50_deploy.prototxt"
compiler.weights = "../models/caffe/resnet/fp32/resnet50.caffemodel"
compiler.memory = 3
compiler.dsp = 28

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


# As can be seen from the op and transfer counts, Resnet-50 takes about 2x the number of ops (multiply + add operations) and because we gave the compiler less on-chip memory, it now enables DDR transfers to move certain intermediate results to and from DDR.
# 
# Well done! That concludes Part 1. Continue on to Part 2: 
# 
# ## [**Part 2:** Using the xfDNN Quantizer to Recalibrate Models][]   
# [**Part 2:** Using the xfDNN Quantizer to Recalibrate Models]: quantizer_caffe.ipynb
