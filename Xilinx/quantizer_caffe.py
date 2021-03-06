
# coding: utf-8

# # Part 2: Using the xfDNN Quantizer to Recalibrate Models
# 
# ## Introduction 
# 
# In this part of the lab, we will look at quantizing 32-bit floating point models to Int16 or Int8 inpreparation for deployment. Deploying Int16/8 models dramatically improves inference deployment and lowers latency. While floating point precision is useful in model training, it is more energy efficient as well as lower latency to deploy models in lower precison. 
# 
# The xfDNN Quantizer performs a technique of quantization known as recalibration. This technique does not require full retraining of the model, and can be accomplished in a matter of seconds, as you will see below. It also allows you to maintain the accuracy of the high precision model.
# 
# Quantization of the model does not alter the orginal high precision model, rather, it calculates the dynamic range of the model and produces scaling parameters recorded in a json file, which will be used by the xDNN overlay during execution of the network/model. Quantization of the model is an offline process that only needs to be performed once per model. The quantizer produces an optimal target quantization from a given network (prototxt and caffemodel) and calibration set (unlabeled input images) without requiring hours of retraining or a labeled dataset.
# 
# In this lab, we will look at quantizing an optimized model generated from Part 1, defined in Caffe prototxt and caffemodel, to Int16 and Int8.  Depending on your earlier notebook this will be either a GoogLeNet-v1 or Resnet-50 model.
# 
# Just like in Part 1, first we will run through an example, then you will get a chance to try the quantizer yourself. 
# 
# ### 1. Import required packages 

# In[1]:


import os,sys
from __future__ import print_function

# Bring in Xilinx ML-Suite Compiler
from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer


# ### 1. Create Quantizer Instance and run it
# 
# To simplify handling of arguments, a config dictionary is used. Take a look at the dictionary below.
# 
# The arguments that need to be passed are:
# - `outmodel` - Filename generated by the compiler for the optimized prototxt and caffemodel.
# - `quantizecfg` - Output JSON filename of quantization scaling parameters. 
# - `bitwidths` - Desired precision from quantizer. This is to set the precision for [image data, weight bitwidth, conv output]. All three values need to be set to the same setting. The valid options are `16` for Int16 and `8` for Int8.  
# - `in_shape` - Sets the desired input image size of the first layer. Images will be resized to these demensions and must match the network data/placeholder layer.
# - `transpose` - Images start as H,W,C (H=0,W=1,C=2) transpose swaps to C,H,W (2,0,1) for typical networks.
# - `channel_swap` - Depending on network training and image read, can swap from RGB (R=0,G=1,B=2) to BGR (2,1,0).
# - `raw_scale` - Depending on network training, scale pixel values before mean subtraction.
# - `img_mean` - Depending on network training, subtract image mean if available.
# - `input_scale` - Depending on network training, scale after subtracting mean.
# - `calibration_size` - Number of images the quantizer will use to calculate the dynamic range. 
# - `calibration_directory` - Location of dir of images used for the calibration process. 
# 
# Below is an example with all the parameters filled in. `channel_swap` `raw_scale` `img_mean` `input_scale` are expert parameters that should be left in the default positions, indicated below. 

# In[2]:


# Use a config dictionary to pass parameters to the compiler
config = {}

config["caffemodel"] = "work/optimized_model" # String for naming intermediate prototxt, caffemodel

# Quantizer Arguments
#config["outmodel"] = Defined in Step 1 # String for naming intermediate prototxt, caffemodel
config["quantizecfg"] = "work/quantization_params.json" # Quantizer will generate quantization params
config["bitwidths"] = [16,16,16] # Supported quantization precision
config["in_shape"] = [3,224,224] # Images will be resized to this shape -> Needs to match prototxt
config["transpose"] = [2,0,1] # (H,W,C)->(C,H,W) transpose argument to quantizer
config["channel_swap"] = [2,1,0] # (R,G,B)->(B,G,R) Channel Swap argument to quantizer
config["raw_scale"] = 255.0
config["img_mean"] = [104.007, 116.669, 122.679] # Mean of the training set (From Imagenet)
config["input_scale"] = 1.0
config["calibration_size"] = 8 # Number of calibration images quantizer will use
config["calibration_directory"] = "../xfdnn/tools/quantize/calibration_directory" # Directory of images

quantizer = xfdnnQuantizer(
    deploy_model=config["caffemodel"]+".prototxt",        # Model filename: input file
    weights=config["caffemodel"]+".caffemodel",           # Floating Point weights
    output_json=config["quantizecfg"],                    # Quantization JSON output filename
    bitwidths=config["bitwidths"],                        # Fixed Point precision: 8,8,8 or 16,16,16
    dims=config["in_shape"],                              # Image dimensions [C,H,W]
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


# ### 2. Try it yourself by changing the quantization precision
# 
# Now that you have had a chance to see how this works, it's time to get some hands on experience.  
# Change the following from the example above:
# 1. Precision of quantization by adjusting `bitwidth`
# 
# Below, replace `value` with one of the supported precision types. [8,8,8] or [16,16,16]

# In[3]:


# Since we already have an instance of the quantizer, you can just update these params:

quantizer.bitwidths = [8,8,8]

# Invoke quantizer
try:
    quantizer.quantize()

    import json
    data = json.loads(open(config["quantizecfg"]).read())
    print("**********\nSuccessfully produced quantization JSON file for %d layers.\n"%len(data['network']))
except Exception as e:
    print("Failed to quantize:",e)


# Well done! That concludes the Part 2. Now you are ready to put parts 1 and 2 together and deploy a network/model. 
# 
# ## [Part 3: Putting it all together: Compile, Quantize and Deploy][]
# 
# [Part 3: Putting it all together: Compile, Quantize and Deploy]: image_classification_caffe.ipynb
