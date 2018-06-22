
Part 1: Using the xfDNN Compiler
================================

Introduction
------------

In this part, you will learn what steps are required to prepare and
compile a network and model. Before being able to deploy networks/models
to Xilinx FPGAs you will need to compile them.

This step is more than just converting the framework graph
representation to one Xilinx can execute. The xfDNN Compiler is a high
performance optimizer for Machine Learning inference. Some of the
techniques it performs are fusing and merging layers, optimizing memory
usage and pre-scheduling complete network deployment. These techniques
increase inference rates and lower inference latency.

Using the xfDNN Compiler is an offline process, which only needs to be
performed once per network. As you will see, the process is simple and
quick.

First, we will look at an example already ready to be run based on
Inception v1. Lets look at each step:

1. Import the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    import os,sys
    from __future__ import print_function
    
    # Bring in Xilinx ML-Suite Compiler
    from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend as xfdnnCompiler

2. Define a new xfdnnCompiler instance and pass arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To simplify handling of arguments, a config dictionary is used. Take a
look at the dictionary below.

The arguments that need to be passed are: - ``prototxt`` - Caffe
representation of the network - ``caffemodel`` - Pre-trained Model for
the network - ``outmodel`` - Filename to save the prototxt of the
optimized network - ``fpgacommands`` - Filename to save
micro-instruction produced by the compiler needed to deploy - ``memory``
- Parameter to set the on-chip memory for the target xDNN overlay. This
example will target an overlay with 5 MB of cache. - ``dsp`` - Parameter
to set the size of the target xDNN overlay. This example uses an overlay
of size 56x32 DSPs. - ``ddr`` - Amount of off-chip memory available.
This example will allow the compiler to use up to 256 MB on DDR.

The xfDNN Compiler interfaces with Caffe to read a network graph, and
generates a sequence of instructions for the xfDNN Deploy APIs to
execute on the FPGA.

During this process the xfDNN Compiler performs computational graph
traversal, node merging and optimization, memory allocation and
optimization and, finally, micro-instruction generation.

.. code:: ipython2

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



.. parsed-literal::

    Network: ../models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt
    GenerateCode: work/bvlc_googlenet_without_lrn.cmd
    Weights: ../models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel
    PngFile: None
    ConcatStrategy: None
    Strategy: all
    ScheduleFile: None
    DSP: 56
    Verbose: False
    FromTF: False
    Memory: 5
    DDR: 256
    Phase: TEST
    RankDir: BT
    
    **************************************************
    * BUILDING DATA FLOW GRAPH
    **************************************************
    
    **************************************************
    * BUILDING NETWORK SCHEDULE
    **************************************************
    Network Schedule ['data', 'conv1/7x7_s2', 'conv1/relu_7x7', 'pool1/3x3_s2', 'conv2/3x3_reduce', 'conv2/relu_3x3_reduce', 'conv2/3x3', 'conv2/relu_3x3', 'pool2/3x3_s2', 'inception_3a/1x1', 'inception_3a/relu_1x1', 'inception_3a/3x3_reduce', 'inception_3a/relu_3x3_reduce', 'inception_3a/3x3', 'inception_3a/relu_3x3', 'inception_3a/5x5_reduce', 'inception_3a/relu_5x5_reduce', 'inception_3a/5x5', 'inception_3a/relu_5x5', 'inception_3a/pool', 'inception_3a/pool_proj', 'inception_3a/relu_pool_proj', 'inception_3a/output', 'inception_3b/1x1', 'inception_3b/relu_1x1', 'inception_3b/3x3_reduce', 'inception_3b/relu_3x3_reduce', 'inception_3b/3x3', 'inception_3b/relu_3x3', 'inception_3b/5x5_reduce', 'inception_3b/relu_5x5_reduce', 'inception_3b/5x5', 'inception_3b/relu_5x5', 'inception_3b/pool', 'inception_3b/pool_proj', 'inception_3b/relu_pool_proj', 'inception_3b/output', 'pool3/3x3_s2', 'inception_4a/1x1', 'inception_4a/relu_1x1', 'inception_4a/3x3_reduce', 'inception_4a/relu_3x3_reduce', 'inception_4a/3x3', 'inception_4a/relu_3x3', 'inception_4a/5x5_reduce', 'inception_4a/relu_5x5_reduce', 'inception_4a/5x5', 'inception_4a/relu_5x5', 'inception_4a/pool', 'inception_4a/pool_proj', 'inception_4a/relu_pool_proj', 'inception_4a/output', 'inception_4b/1x1', 'inception_4b/relu_1x1', 'inception_4b/3x3_reduce', 'inception_4b/relu_3x3_reduce', 'inception_4b/3x3', 'inception_4b/relu_3x3', 'inception_4b/5x5_reduce', 'inception_4b/relu_5x5_reduce', 'inception_4b/5x5', 'inception_4b/relu_5x5', 'inception_4b/pool', 'inception_4b/pool_proj', 'inception_4b/relu_pool_proj', 'inception_4b/output', 'inception_4c/1x1', 'inception_4c/relu_1x1', 'inception_4c/3x3_reduce', 'inception_4c/relu_3x3_reduce', 'inception_4c/3x3', 'inception_4c/relu_3x3', 'inception_4c/5x5_reduce', 'inception_4c/relu_5x5_reduce', 'inception_4c/5x5', 'inception_4c/relu_5x5', 'inception_4c/pool', 'inception_4c/pool_proj', 'inception_4c/relu_pool_proj', 'inception_4c/output', 'inception_4d/1x1', 'inception_4d/relu_1x1', 'inception_4d/3x3_reduce', 'inception_4d/relu_3x3_reduce', 'inception_4d/3x3', 'inception_4d/relu_3x3', 'inception_4d/5x5_reduce', 'inception_4d/relu_5x5_reduce', 'inception_4d/5x5', 'inception_4d/relu_5x5', 'inception_4d/pool', 'inception_4d/pool_proj', 'inception_4d/relu_pool_proj', 'inception_4d/output', 'inception_4e/1x1', 'inception_4e/relu_1x1', 'inception_4e/3x3_reduce', 'inception_4e/relu_3x3_reduce', 'inception_4e/3x3', 'inception_4e/relu_3x3', 'inception_4e/5x5_reduce', 'inception_4e/relu_5x5_reduce', 'inception_4e/5x5', 'inception_4e/relu_5x5', 'inception_4e/pool', 'inception_4e/pool_proj', 'inception_4e/relu_pool_proj', 'inception_4e/output', 'pool4/3x3_s2', 'inception_5a/1x1', 'inception_5a/relu_1x1', 'inception_5a/3x3_reduce', 'inception_5a/relu_3x3_reduce', 'inception_5a/3x3', 'inception_5a/relu_3x3', 'inception_5a/5x5_reduce', 'inception_5a/relu_5x5_reduce', 'inception_5a/5x5', 'inception_5a/relu_5x5', 'inception_5a/pool', 'inception_5a/pool_proj', 'inception_5a/relu_pool_proj', 'inception_5a/output', 'inception_5b/1x1', 'inception_5b/relu_1x1', 'inception_5b/3x3_reduce', 'inception_5b/relu_3x3_reduce', 'inception_5b/3x3', 'inception_5b/relu_3x3', 'inception_5b/5x5_reduce', 'inception_5b/relu_5x5_reduce', 'inception_5b/5x5', 'inception_5b/relu_5x5', 'inception_5b/pool', 'inception_5b/pool_proj', 'inception_5b/relu_pool_proj', 'inception_5b/output', 'pool5/7x7_s1', 'pool5/drop_7x7_s1', 'loss3/classifier', 'prob']
    
    **************************************************
    * COMPUTING MEMORY REQUIREMENTS
    **************************************************
    Minimum Memory __________
    33 ['inception_3b/pool'] size:3325952.0 remap:[] data movement:[]
    33	inception_3a/output_blob M[0,917504] Z=917504 F=[23, 25, 29, 33] B=[22] E=[] S=['replace_layer'] ['concat'] L=-1
    33	inception_3b/pool_blob M[0,917504] Z=917504 F=[34] B=[33] E=[] S=['layer'] [] L=-1
    33	inception_3b/5x5_blob M[0,344064] Z=344064 F=[32, 36] B=[31, 32] E=[] S=['layer'] ['concat'] L=-1
    33	inception_3b/3x3_blob M[0,688128] Z=688128 F=[28, 36] B=[27, 28] E=[] S=['layer'] ['concat'] L=-1
    33	inception_3b/1x1_blob M[0,458752] Z=458752 F=[24, 36] B=[23, 24] E=[] S=['layer'] ['concat'] L=-1
    
    **************************************************
    * ALLOCATING DYNAMIC MEMORY SCHEDULE
    **************************************************
    Allocating Memory
    Trying strategy bysize
    
    **************************************************
    * GENERATING OUTPUT REPORTS
    **************************************************
    Minimum Memory 105 ['inception_4e/pool_proj'] 4300800.0
    inception_4e/pool_blob M[3354624,4300800] Z=946176 F=[105] B=[104] E=[1] S=['layer'] [] L=0
    inception_4e/1x1_blob M[1863680,2322432] Z=458752 F=[95, 107] B=[94, 95] E=[-1] S=['layer'] ['concat'] L=0
    inception_4e/3x3_blob M[2322432,2895872] Z=573440 F=[99, 107] B=[98, 99] E=[-1] S=['layer'] ['concat'] L=0
    inception_4e/5x5_blob M[2895872,3125248] Z=229376 F=[103, 107] B=[102, 103] E=[-1] S=['layer'] ['concat'] L=0
    inception_4e/pool_proj_blob M[3125248,3354624] Z=229376 F=[106, 107] B=[105, 106] E=[-1] S=['layer'] ['concat'] L=0
    
    **************************************************
    * GENERATING OUTPUT FILES
    **************************************************
    XDNN Command file: work/bvlc_googlenet_without_lrn.cmd
    XDNN JSON Report file: work/bvlc_googlenet_without_lrn.cmd.json
    OUTPUT REPORT:
    Unsupported Layers: 3
    0) loss3/classifier
    	Attributes: ('## 140 XNInner loss3/classifier 16 26 2 0x0 1 1024 0x20000 1000 1000', u'loss3/classifier: type=InnerProduct, sizes=None, shapes=[[1000, 1024], [1000]], sched 139 Kernel None Strides None Padding None  NO VALID CODE  ')
    1) data
    	Attributes: ("# LAYER data [u'Input'] ['layer']", u'data: type=Input, sizes=None, shapes=None, sched 0 Kernel None Strides None Padding None  NO VALID CODE  ')
    2) prob
    	Attributes: ("# LAYER prob [u'Softmax'] ['layer']", u'prob: type=Softmax, sizes=None, shapes=None, sched 140 Kernel None Strides None Padding None  NO VALID CODE  ')
    Compiling weights from: ../models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel
    Writing weights to directory work/bvlc_googlenet_without_lrn.caffemodel_data
    SUCCESS
    
    *************************************************
    * GENERATING new PROTO and new caffemodel weights  
    ***************************************************
    conv1/7x7_s2 0: (64, 3, 7, 7) (64, 3, 7, 7)
    conv2/3x3_reduce 0: (64, 64, 1, 1) (64, 64, 1, 1)
    conv2/3x3 0: (192, 64, 3, 3) (192, 64, 3, 3)
    inception_3a/1x1 0: (64, 192, 1, 1) (64, 192, 1, 1)
    inception_3a/3x3_reduce 0: (96, 192, 1, 1) (96, 192, 1, 1)
    inception_3a/3x3 0: (128, 96, 3, 3) (128, 96, 3, 3)
    inception_3a/5x5_reduce 0: (16, 192, 1, 1) (16, 192, 1, 1)
    inception_3a/5x5 0: (32, 16, 5, 5) (32, 16, 5, 5)
    inception_3a/pool_proj 0: (32, 192, 1, 1) (32, 192, 1, 1)
    inception_3b/1x1 0: (128, 256, 1, 1) (128, 256, 1, 1)
    inception_3b/3x3_reduce 0: (128, 256, 1, 1) (128, 256, 1, 1)
    inception_3b/3x3 0: (192, 128, 3, 3) (192, 128, 3, 3)
    inception_3b/5x5_reduce 0: (32, 256, 1, 1) (32, 256, 1, 1)
    inception_3b/5x5 0: (96, 32, 5, 5) (96, 32, 5, 5)
    inception_3b/pool_proj 0: (64, 256, 1, 1) (64, 256, 1, 1)
    inception_4a/1x1 0: (192, 480, 1, 1) (192, 480, 1, 1)
    inception_4a/3x3_reduce 0: (96, 480, 1, 1) (96, 480, 1, 1)
    inception_4a/3x3 0: (208, 96, 3, 3) (208, 96, 3, 3)
    inception_4a/5x5_reduce 0: (16, 480, 1, 1) (16, 480, 1, 1)
    inception_4a/5x5 0: (48, 16, 5, 5) (48, 16, 5, 5)
    inception_4a/pool_proj 0: (64, 480, 1, 1) (64, 480, 1, 1)
    inception_4b/1x1 0: (160, 512, 1, 1) (160, 512, 1, 1)
    inception_4b/3x3_reduce 0: (112, 512, 1, 1) (112, 512, 1, 1)
    inception_4b/3x3 0: (224, 112, 3, 3) (224, 112, 3, 3)
    inception_4b/5x5_reduce 0: (24, 512, 1, 1) (24, 512, 1, 1)
    inception_4b/5x5 0: (64, 24, 5, 5) (64, 24, 5, 5)
    inception_4b/pool_proj 0: (64, 512, 1, 1) (64, 512, 1, 1)
    inception_4c/1x1 0: (128, 512, 1, 1) (128, 512, 1, 1)
    inception_4c/3x3_reduce 0: (128, 512, 1, 1) (128, 512, 1, 1)
    inception_4c/3x3 0: (256, 128, 3, 3) (256, 128, 3, 3)
    inception_4c/5x5_reduce 0: (24, 512, 1, 1) (24, 512, 1, 1)
    inception_4c/5x5 0: (64, 24, 5, 5) (64, 24, 5, 5)
    inception_4c/pool_proj 0: (64, 512, 1, 1) (64, 512, 1, 1)
    inception_4d/1x1 0: (112, 512, 1, 1) (112, 512, 1, 1)
    inception_4d/3x3_reduce 0: (144, 512, 1, 1) (144, 512, 1, 1)
    inception_4d/3x3 0: (288, 144, 3, 3) (288, 144, 3, 3)
    inception_4d/5x5_reduce 0: (32, 512, 1, 1) (32, 512, 1, 1)
    inception_4d/5x5 0: (64, 32, 5, 5) (64, 32, 5, 5)
    inception_4d/pool_proj 0: (64, 512, 1, 1) (64, 512, 1, 1)
    inception_4e/1x1 0: (256, 528, 1, 1) (256, 528, 1, 1)
    inception_4e/3x3_reduce 0: (160, 528, 1, 1) (160, 528, 1, 1)
    inception_4e/3x3 0: (320, 160, 3, 3) (320, 160, 3, 3)
    inception_4e/5x5_reduce 0: (32, 528, 1, 1) (32, 528, 1, 1)
    inception_4e/5x5 0: (128, 32, 5, 5) (128, 32, 5, 5)
    inception_4e/pool_proj 0: (128, 528, 1, 1) (128, 528, 1, 1)
    inception_5a/1x1 0: (256, 832, 1, 1) (256, 832, 1, 1)
    inception_5a/3x3_reduce 0: (160, 832, 1, 1) (160, 832, 1, 1)
    inception_5a/3x3 0: (320, 160, 3, 3) (320, 160, 3, 3)
    inception_5a/5x5_reduce 0: (32, 832, 1, 1) (32, 832, 1, 1)
    inception_5a/5x5 0: (128, 32, 5, 5) (128, 32, 5, 5)
    inception_5a/pool_proj 0: (128, 832, 1, 1) (128, 832, 1, 1)
    inception_5b/1x1 0: (384, 832, 1, 1) (384, 832, 1, 1)
    inception_5b/3x3_reduce 0: (192, 832, 1, 1) (192, 832, 1, 1)
    inception_5b/3x3 0: (384, 192, 3, 3) (384, 192, 3, 3)
    inception_5b/5x5_reduce 0: (48, 832, 1, 1) (48, 832, 1, 1)
    inception_5b/5x5 0: (128, 48, 5, 5) (128, 48, 5, 5)
    inception_5b/pool_proj 0: (128, 832, 1, 1) (128, 832, 1, 1)
    Compiler successfully generated JSON and the data directory: work/bvlc_googlenet_without_lrn.caffemodel_data
    **********
    Compilation Successful!
    
    Network Operations Count: 3176103168
    DDR Transfers (bytes): 0


3. Try it yourself with a different model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have had a chance to see how this works, it's time to get
some hands on experience. Change the following from the example above:
1. The Network - From Inception v1 to ResNet50. 2. The Weights - New
Network, will require to us to re-extract the weights. 3. The amount of
on-chip memory available. 4. The size of the xDNN overlay.

| Resources: ResNet50 prototxt located here:
  ``"../models/caffe/resnet/fp32/resnet50_deploy.prototxt"``
| ResNet50 caffemodel located here:
  ``"../models/caffe/resnet/fp32/resnet50.caffemodel"``

In the last example, you ran through the compiler using a setting of 5
MB. Try ``3`` for 3 MB of on-chip memory. This will restrict the memory
available to the compiler and it will automatically create DDR transfer
commands to move intermediate results between the FPGA and DDR. You will
see the DDR transfers is no longer 0 bytes.

Lastly, 56 or 28 are supported as the DSP argument, so change this to
28. This corresponds to the 28x32 DSP configuration which reduces the
number of DSPs compared to 56x32, but allows us to run twice as many CNN
processing engines.

Note: Because Resnet-50 has more opportunities to optimize the graph,
and the model parameters are approximately 2x the size of the
GoogLeVet-v1 model, it will take slightly longer to write the optimized
weights run compared to the previous GoogLeNet-v1 example.

.. code:: ipython2

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


.. parsed-literal::

    Network: ../models/caffe/resnet/fp32/resnet50_deploy.prototxt
    GenerateCode: work/bvlc_googlenet_without_lrn.cmd
    Weights: ../models/caffe/resnet/fp32/resnet50.caffemodel
    PngFile: None
    ConcatStrategy: None
    Strategy: all
    ScheduleFile: None
    DSP: 28
    Verbose: False
    FromTF: False
    Memory: 3
    DDR: 256
    Phase: 1
    Unknown phase: 1
    RankDir: BT
    
    **************************************************
    * BUILDING DATA FLOW GRAPH
    **************************************************
    
    **************************************************
    * BUILDING NETWORK SCHEDULE
    **************************************************
    Network Schedule ['data', 'conv1', 'bn_conv1', 'scale_conv1', 'conv1_relu', 'pool1', 'res2a_branch1', 'bn2a_branch1', 'scale2a_branch1', 'res2a_branch2a', 'bn2a_branch2a', 'scale2a_branch2a', 'res2a_branch2a_relu', 'res2a_branch2b', 'bn2a_branch2b', 'scale2a_branch2b', 'res2a_branch2b_relu', 'res2a_branch2c', 'bn2a_branch2c', 'scale2a_branch2c', 'res2a', 'res2a_relu', 'res2b_branch2a', 'bn2b_branch2a', 'scale2b_branch2a', 'res2b_branch2a_relu', 'res2b_branch2b', 'bn2b_branch2b', 'scale2b_branch2b', 'res2b_branch2b_relu', 'res2b_branch2c', 'bn2b_branch2c', 'scale2b_branch2c', 'res2b', 'res2b_relu', 'res2c_branch2a', 'bn2c_branch2a', 'scale2c_branch2a', 'res2c_branch2a_relu', 'res2c_branch2b', 'bn2c_branch2b', 'scale2c_branch2b', 'res2c_branch2b_relu', 'res2c_branch2c', 'bn2c_branch2c', 'scale2c_branch2c', 'res2c', 'res2c_relu', 'res3a_branch1', 'bn3a_branch1', 'scale3a_branch1', 'res3a_branch2a', 'bn3a_branch2a', 'scale3a_branch2a', 'res3a_branch2a_relu', 'res3a_branch2b', 'bn3a_branch2b', 'scale3a_branch2b', 'res3a_branch2b_relu', 'res3a_branch2c', 'bn3a_branch2c', 'scale3a_branch2c', 'res3a', 'res3a_relu', 'res3b_branch2a', 'bn3b_branch2a', 'scale3b_branch2a', 'res3b_branch2a_relu', 'res3b_branch2b', 'bn3b_branch2b', 'scale3b_branch2b', 'res3b_branch2b_relu', 'res3b_branch2c', 'bn3b_branch2c', 'scale3b_branch2c', 'res3b', 'res3b_relu', 'res3c_branch2a', 'bn3c_branch2a', 'scale3c_branch2a', 'res3c_branch2a_relu', 'res3c_branch2b', 'bn3c_branch2b', 'scale3c_branch2b', 'res3c_branch2b_relu', 'res3c_branch2c', 'bn3c_branch2c', 'scale3c_branch2c', 'res3c', 'res3c_relu', 'res3d_branch2a', 'bn3d_branch2a', 'scale3d_branch2a', 'res3d_branch2a_relu', 'res3d_branch2b', 'bn3d_branch2b', 'scale3d_branch2b', 'res3d_branch2b_relu', 'res3d_branch2c', 'bn3d_branch2c', 'scale3d_branch2c', 'res3d', 'res3d_relu', 'res4a_branch1', 'bn4a_branch1', 'scale4a_branch1', 'res4a_branch2a', 'bn4a_branch2a', 'scale4a_branch2a', 'res4a_branch2a_relu', 'res4a_branch2b', 'bn4a_branch2b', 'scale4a_branch2b', 'res4a_branch2b_relu', 'res4a_branch2c', 'bn4a_branch2c', 'scale4a_branch2c', 'res4a', 'res4a_relu', 'res4b_branch2a', 'bn4b_branch2a', 'scale4b_branch2a', 'res4b_branch2a_relu', 'res4b_branch2b', 'bn4b_branch2b', 'scale4b_branch2b', 'res4b_branch2b_relu', 'res4b_branch2c', 'bn4b_branch2c', 'scale4b_branch2c', 'res4b', 'res4b_relu', 'res4c_branch2a', 'bn4c_branch2a', 'scale4c_branch2a', 'res4c_branch2a_relu', 'res4c_branch2b', 'bn4c_branch2b', 'scale4c_branch2b', 'res4c_branch2b_relu', 'res4c_branch2c', 'bn4c_branch2c', 'scale4c_branch2c', 'res4c', 'res4c_relu', 'res4d_branch2a', 'bn4d_branch2a', 'scale4d_branch2a', 'res4d_branch2a_relu', 'res4d_branch2b', 'bn4d_branch2b', 'scale4d_branch2b', 'res4d_branch2b_relu', 'res4d_branch2c', 'bn4d_branch2c', 'scale4d_branch2c', 'res4d', 'res4d_relu', 'res4e_branch2a', 'bn4e_branch2a', 'scale4e_branch2a', 'res4e_branch2a_relu', 'res4e_branch2b', 'bn4e_branch2b', 'scale4e_branch2b', 'res4e_branch2b_relu', 'res4e_branch2c', 'bn4e_branch2c', 'scale4e_branch2c', 'res4e', 'res4e_relu', 'res4f_branch2a', 'bn4f_branch2a', 'scale4f_branch2a', 'res4f_branch2a_relu', 'res4f_branch2b', 'bn4f_branch2b', 'scale4f_branch2b', 'res4f_branch2b_relu', 'res4f_branch2c', 'bn4f_branch2c', 'scale4f_branch2c', 'res4f', 'res4f_relu', 'res5a_branch1', 'bn5a_branch1', 'scale5a_branch1', 'res5a_branch2a', 'bn5a_branch2a', 'scale5a_branch2a', 'res5a_branch2a_relu', 'res5a_branch2b', 'bn5a_branch2b', 'scale5a_branch2b', 'res5a_branch2b_relu', 'res5a_branch2c', 'bn5a_branch2c', 'scale5a_branch2c', 'res5a', 'res5a_relu', 'res5b_branch2a', 'bn5b_branch2a', 'scale5b_branch2a', 'res5b_branch2a_relu', 'res5b_branch2b', 'bn5b_branch2b', 'scale5b_branch2b', 'res5b_branch2b_relu', 'res5b_branch2c', 'bn5b_branch2c', 'scale5b_branch2c', 'res5b', 'res5b_relu', 'res5c_branch2a', 'bn5c_branch2a', 'scale5c_branch2a', 'res5c_branch2a_relu', 'res5c_branch2b', 'bn5c_branch2b', 'scale5c_branch2b', 'res5c_branch2b_relu', 'res5c_branch2c', 'bn5c_branch2c', 'scale5c_branch2c', 'res5c', 'res5c_relu', 'pool5', 'fc1000', 'prob']
    
    **************************************************
    * COMPUTING MEMORY REQUIREMENTS
    **************************************************
    Minimum Memory __________
    17 ['res2a_branch2c'] size:4128768.0 remap:[] data movement:[]
    17	res2a_branch2b_blob M[0,458752] Z=458752 F=[14, 15, 16, 17] B=[13, 14, 15, 16] E=[] S=['layer'] [] L=-1
    17	res2a_branch2c_blob M[0,1835008] Z=1835008 F=[18, 19, 20] B=[17, 18, 19] E=[] S=['layer'] [] L=-1
    17	res2a_branch1_blob M[0,1835008] Z=1835008 F=[7, 8, 20] B=[6, 7, 8] E=[] S=['layer'] [] L=-1
    
    **************************************************
    * ALLOCATING DYNAMIC MEMORY SCHEDULE
    **************************************************
    Allocating Memory
    Trying strategy bysize
    
    **************************************************
    * GENERATING OUTPUT REPORTS
    **************************************************
    Minimum Memory 46 ['res2c'] 3670016.0
    res2c_branch2c_blob M[1835008,3670016] Z=1835008 F=[44, 45, 46] B=[43, 44, 45] E=[1] S=['layer'] [] L=1
    res2c_blob M[0,1835008] Z=1835008 F=[47, 48, 51] B=[46, 47] E=[1] S=['replace_layer'] [] L=0
    
    **************************************************
    * GENERATING OUTPUT FILES
    **************************************************
    XDNN Command file: work/bvlc_googlenet_without_lrn.cmd
    XDNN JSON Report file: work/bvlc_googlenet_without_lrn.cmd.json
    OUTPUT REPORT:
    Unsupported Layers: 3
    0) fc1000
    	Attributes: ('## 261 XNInner fc1000 16 26 2 0x0 1 2048 0x20000 1000 1000', u'fc1000: type=InnerProduct, sizes=None, shapes=[[1000, 2048], [1000]], sched 227 Kernel None Strides None Padding None  NO VALID CODE  ')
    1) data
    	Attributes: ("# LAYER data [u'Input'] ['layer']", u'data: type=Input, sizes=None, shapes=None, sched 0 Kernel None Strides None Padding None  NO VALID CODE  ')
    2) prob
    	Attributes: ("# LAYER prob [u'Softmax'] ['layer']", u'prob: type=Softmax, sizes=None, shapes=None, sched 228 Kernel None Strides None Padding None  NO VALID CODE  ')
    Compiling weights from: ../models/caffe/resnet/fp32/resnet50.caffemodel
    Writing weights to directory work/resnet50.caffemodel_data
    SUCCESS
    
    *************************************************
    * GENERATING new PROTO and new caffemodel weights  
    ***************************************************
    conv1 0: (64, 3, 7, 7) (64, 3, 7, 7)
    res2a_branch1 0: (256, 64, 1, 1) (256, 64, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res2a_branch2a 0: (64, 64, 1, 1) (64, 64, 1, 1)
    bias shape (64,)
    1: (64,) (64,)
    res2a_branch2b 0: (64, 64, 3, 3) (64, 64, 3, 3)
    bias shape (64,)
    1: (64,) (64,)
    res2a_branch2c 0: (256, 64, 1, 1) (256, 64, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res2b_branch2a 0: (64, 256, 1, 1) (64, 256, 1, 1)
    bias shape (64,)
    1: (64,) (64,)
    res2b_branch2b 0: (64, 64, 3, 3) (64, 64, 3, 3)
    bias shape (64,)
    1: (64,) (64,)
    res2b_branch2c 0: (256, 64, 1, 1) (256, 64, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res2c_branch2a 0: (64, 256, 1, 1) (64, 256, 1, 1)
    bias shape (64,)
    1: (64,) (64,)
    res2c_branch2b 0: (64, 64, 3, 3) (64, 64, 3, 3)
    bias shape (64,)
    1: (64,) (64,)
    res2c_branch2c 0: (256, 64, 1, 1) (256, 64, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res3a_branch1 0: (512, 256, 1, 1) (512, 256, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res3a_branch2a 0: (128, 256, 1, 1) (128, 256, 1, 1)
    bias shape (128,)
    1: (128,) (128,)
    res3a_branch2b 0: (128, 128, 3, 3) (128, 128, 3, 3)
    bias shape (128,)
    1: (128,) (128,)
    res3a_branch2c 0: (512, 128, 1, 1) (512, 128, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res3b_branch2a 0: (128, 512, 1, 1) (128, 512, 1, 1)
    bias shape (128,)
    1: (128,) (128,)
    res3b_branch2b 0: (128, 128, 3, 3) (128, 128, 3, 3)
    bias shape (128,)
    1: (128,) (128,)
    res3b_branch2c 0: (512, 128, 1, 1) (512, 128, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res3c_branch2a 0: (128, 512, 1, 1) (128, 512, 1, 1)
    bias shape (128,)
    1: (128,) (128,)
    res3c_branch2b 0: (128, 128, 3, 3) (128, 128, 3, 3)
    bias shape (128,)
    1: (128,) (128,)
    res3c_branch2c 0: (512, 128, 1, 1) (512, 128, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res3d_branch2a 0: (128, 512, 1, 1) (128, 512, 1, 1)
    bias shape (128,)
    1: (128,) (128,)
    res3d_branch2b 0: (128, 128, 3, 3) (128, 128, 3, 3)
    bias shape (128,)
    1: (128,) (128,)
    res3d_branch2c 0: (512, 128, 1, 1) (512, 128, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res4a_branch1 0: (1024, 512, 1, 1) (1024, 512, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4a_branch2a 0: (256, 512, 1, 1) (256, 512, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4a_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4a_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4b_branch2a 0: (256, 1024, 1, 1) (256, 1024, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4b_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4b_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4c_branch2a 0: (256, 1024, 1, 1) (256, 1024, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4c_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4c_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4d_branch2a 0: (256, 1024, 1, 1) (256, 1024, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4d_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4d_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4e_branch2a 0: (256, 1024, 1, 1) (256, 1024, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4e_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4e_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res4f_branch2a 0: (256, 1024, 1, 1) (256, 1024, 1, 1)
    bias shape (256,)
    1: (256,) (256,)
    res4f_branch2b 0: (256, 256, 3, 3) (256, 256, 3, 3)
    bias shape (256,)
    1: (256,) (256,)
    res4f_branch2c 0: (1024, 256, 1, 1) (1024, 256, 1, 1)
    bias shape (1024,)
    1: (1024,) (1024,)
    res5a_branch1 0: (2048, 1024, 1, 1) (2048, 1024, 1, 1)
    bias shape (2048,)
    1: (2048,) (2048,)
    res5a_branch2a 0: (512, 1024, 1, 1) (512, 1024, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res5a_branch2b 0: (512, 512, 3, 3) (512, 512, 3, 3)
    bias shape (512,)
    1: (512,) (512,)
    res5a_branch2c 0: (2048, 512, 1, 1) (2048, 512, 1, 1)
    bias shape (2048,)
    1: (2048,) (2048,)
    res5b_branch2a 0: (512, 2048, 1, 1) (512, 2048, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res5b_branch2b 0: (512, 512, 3, 3) (512, 512, 3, 3)
    bias shape (512,)
    1: (512,) (512,)
    res5b_branch2c 0: (2048, 512, 1, 1) (2048, 512, 1, 1)
    bias shape (2048,)
    1: (2048,) (2048,)
    res5c_branch2a 0: (512, 2048, 1, 1) (512, 2048, 1, 1)
    bias shape (512,)
    1: (512,) (512,)
    res5c_branch2b 0: (512, 512, 3, 3) (512, 512, 3, 3)
    bias shape (512,)
    1: (512,) (512,)
    res5c_branch2c 0: (2048, 512, 1, 1) (2048, 512, 1, 1)
    bias shape (2048,)
    1: (2048,) (2048,)
    Compiler successfully generated JSON and the data directory: work/bvlc_googlenet_without_lrn.caffemodel_data
    **********
    Compilation Successful!
    
    Network Operations Count: 7719276544
    DDR Transfers (bytes): 33030144


As can be seen from the op and transfer counts, Resnet-50 takes about 2x
the number of ops (multiply + add operations) and because we gave the
compiler less on-chip memory, it now enables DDR transfers to move
certain intermediate results to and from DDR.

Well done! That concludes Part 1. Continue on to Part 2:

`**Part 2:** Using the xfDNN Quantizer to Recalibrate Models <quantizer_caffe.ipynb>`__
---------------------------------------------------------------------------------------
