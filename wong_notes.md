## Assignment 1


## Assignment 2
>**Question1.1:**
>Train the model using LoRA instruction above and monitor the change in loss during training. Does it behave as expected?  Note: Since the entire training process may take a long time, press `Ctrl+c` to stop monitoring once you have gathered enough information.

The loss gradually drops but not consistently
<details>
<summary><strong>Settings</strong></summary>

```bash
MODEL_PATH="/public/Meta-Llama-3-8B-Instruct"

python finetune.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --report_to tensorboard \
    --group_by_length \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --load_in_8bit \
    --use_peft \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_alpha 16 \
    --log_level info \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 414 \
    --save_steps 414 \

```
```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000001:00:00.0 Off |                    0 |
| N/A   66C    P0            290W /  300W |   30833MiB /  81920MiB |     93%      Default |
|                                         |                        |             Disabled
```
</details>

<details>
<summary><strong>Loss output</strong></summary>

```bash
Output:
{'loss': 0.8633, 'grad_norm': 3.3755338191986084, 'learning_rate': 1.4046822742474915e-05, 'epoch': 0.0}                                                                                                                                                                                        
{'loss': 1.1785, 'grad_norm': 3.5265350341796875, 'learning_rate': 1.5050167224080267e-05, 'epoch': 0.0}                                                                                                                                                                                        
{'loss': 0.4775, 'grad_norm': 2.3630809783935547, 'learning_rate': 1.6053511705685617e-05, 'epoch': 0.0}                                                                                                                                                                                        
{'loss': 0.5543, 'grad_norm': 3.203991651535034, 'learning_rate': 1.7056856187290967e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.792, 'grad_norm': 2.9750521183013916, 'learning_rate': 1.806020066889632e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.5296, 'grad_norm': 2.8203935623168945, 'learning_rate': 1.906354515050167e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.6645, 'grad_norm': 3.0078506469726562, 'learning_rate': 2.006688963210702e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.4368, 'grad_norm': 3.0042645931243896, 'learning_rate': 2.107023411371237e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.3474, 'grad_norm': 2.319020986557007, 'learning_rate': 2.2073578595317725e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.6491, 'grad_norm': 1.8045371770858765, 'learning_rate': 2.3076923076923076e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 0.4396, 'grad_norm': 2.137944459915161, 'learning_rate': 2.4080267558528427e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.4046, 'grad_norm': 1.8396389484405518, 'learning_rate': 2.5083612040133777e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 0.5885, 'grad_norm': 2.218198299407959, 'learning_rate': 2.6086956521739128e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.5067, 'grad_norm': 2.432325839996338, 'learning_rate': 2.709030100334448e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.3572, 'grad_norm': 1.6359463930130005, 'learning_rate': 2.809364548494983e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.6819, 'grad_norm': 2.9594814777374268, 'learning_rate': 2.9096989966555184e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 1.078, 'grad_norm': 4.268344402313232, 'learning_rate': 3.0100334448160535e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.2473, 'grad_norm': 1.4610995054244995, 'learning_rate': 3.1103678929765886e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 0.4482, 'grad_norm': 2.285464286804199, 'learning_rate': 3.210702341137123e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.5643, 'grad_norm': 1.964709758758545, 'learning_rate': 3.311036789297659e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.4297, 'grad_norm': 2.9744064807891846, 'learning_rate': 3.4113712374581935e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 0.5995, 'grad_norm': 3.8179707527160645, 'learning_rate': 3.511705685618729e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.2549, 'grad_norm': 2.347892999649048, 'learning_rate': 3.612040133779264e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.493, 'grad_norm': 4.911048889160156, 'learning_rate': 3.712374581939799e-05, 'epoch': 0.01}                                                                                                                                                                                          
{'loss': 0.0871, 'grad_norm': 2.1859829425811768, 'learning_rate': 3.812709030100334e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.2495, 'grad_norm': 2.8492250442504883, 'learning_rate': 3.913043478260869e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.1135, 'grad_norm': 1.7983886003494263, 'learning_rate': 4.013377926421404e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.3571, 'grad_norm': 2.487802743911743, 'learning_rate': 4.1137123745819394e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.2487, 'grad_norm': 4.6402268409729, 'learning_rate': 4.214046822742474e-05, 'epoch': 0.01}                                                                                                                                                                                           
{'loss': 0.4219, 'grad_norm': 3.3043532371520996, 'learning_rate': 4.3143812709030096e-05, 'epoch': 0.01}                                                                                                                                                                                       
{'loss': 0.2375, 'grad_norm': 1.9599971771240234, 'learning_rate': 4.414715719063545e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.137, 'grad_norm': 0.9967252612113953, 'learning_rate': 4.5150501672240804e-05, 'epoch': 0.01} 
{'loss': 0.5382, 'grad_norm': 2.58703351020813, 'learning_rate': 4.615384615384615e-05, 'epoch': 0.01}                                                                                                                                                                                          
{'loss': 0.5801, 'grad_norm': 3.632885456085205, 'learning_rate': 4.7157190635451506e-05, 'epoch': 0.01}                                                                                                                                                                                        
{'loss': 0.7109, 'grad_norm': 3.399437427520752, 'learning_rate': 4.816053511705685e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.4014, 'grad_norm': 1.8135490417480469, 'learning_rate': 4.91638795986622e-05, 'epoch': 0.01}                                                                                                                                                                                         
{'loss': 0.1059, 'grad_norm': 0.9409568309783936, 'learning_rate': 5.0167224080267555e-05, 'epoch': 0.02}                                                                                                                                                                                       
{'loss': 0.5548, 'grad_norm': 2.3657190799713135, 'learning_rate': 5.11705685618729e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.249, 'grad_norm': 2.254606008529663, 'learning_rate': 5.2173913043478256e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.2759, 'grad_norm': 1.8322852849960327, 'learning_rate': 5.3177257525083604e-05, 'epoch': 0.02}                                                                                                                                                                                       
{'loss': 0.8165, 'grad_norm': 3.375502586364746, 'learning_rate': 5.418060200668896e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.3014, 'grad_norm': 2.870757818222046, 'learning_rate': 5.5183946488294305e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.5726, 'grad_norm': 2.7160677909851074, 'learning_rate': 5.618729096989966e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.1658, 'grad_norm': 0.9706992506980896, 'learning_rate': 5.7190635451505014e-05, 'epoch': 0.02}                                                                                                                                                                                       
{'loss': 0.7327, 'grad_norm': 2.717672109603882, 'learning_rate': 5.819397993311037e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.2006, 'grad_norm': 1.4196524620056152, 'learning_rate': 5.9197324414715715e-05, 'epoch': 0.02}                                                                                                                                                                                       
{'loss': 0.2401, 'grad_norm': 1.640075445175171, 'learning_rate': 6.020066889632107e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.2332, 'grad_norm': 1.4920846223831177, 'learning_rate': 6.120401337792642e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.2157, 'grad_norm': 1.6795705556869507, 'learning_rate': 6.220735785953177e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.2663, 'grad_norm': 2.4266812801361084, 'learning_rate': 6.321070234113711e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.2707, 'grad_norm': 1.8791099786758423, 'learning_rate': 6.421404682274247e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.2517, 'grad_norm': 1.5601716041564941, 'learning_rate': 6.521739130434782e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.3827, 'grad_norm': 1.6370728015899658, 'learning_rate': 6.622073578595317e-05, 'epoch': 0.02}                                                                                                                                                                                        
{'loss': 0.5031, 'grad_norm': 3.986889362335205, 'learning_rate': 6.722408026755852e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.0655, 'grad_norm': 0.764961302280426, 'learning_rate': 6.822742474916387e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.127, 'grad_norm': 1.0843242406845093, 'learning_rate': 6.923076923076922e-05, 'epoch': 0.02}                                                                                                                                                                                         
{'loss': 0.1481, 'grad_norm': 3.4226584434509277, 'learning_rate': 7.023411371237458e-05, 'epoch': 0.02}       

```
</details>

<br>

>**Question1.2:**
>You can see that the default values for `per_device_train_batch_size` and `per_device_eval_batch_size` are currently set to 1, but these can be adjusted for improved performance. Increasing the batch size can accelerate training, although it may lead to out-of-memory errors. Determine the optimal batch size for LoRA training by observing the time taken to train and GPU memory usage with the `nvidia-smi` command in a separate terminal window. Does GPU memory usage change as expected with varying batch sizes? Again, press 'Ctrl+c' to stop monitoring once you have gathered enough information.


The max batch size I could input was 2. 
```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000001:00:00.0 Off |                    0 |
| N/A   64C    P0             90W /  300W |   57181MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
```

GPU memory increases with increasing batch size
<details>
<summary><strong>Loss</strong></summary>

```bash
{'loss': 0.7492, 'grad_norm': 2.8886396884918213, 'learning_rate': 2e-06, 'epoch': 0.0}                                                       
{'loss': 0.8722, 'grad_norm': 2.1094565391540527, 'learning_rate': 4e-06, 'epoch': 0.0}                                                       
{'loss': 1.0622, 'grad_norm': 3.2013118267059326, 'learning_rate': 5.999999999999999e-06, 'epoch': 0.0}                                       
{'loss': 0.7889, 'grad_norm': 3.1231603622436523, 'learning_rate': 8e-06, 'epoch': 0.0}                                                       
{'loss': 0.8092, 'grad_norm': 3.0500237941741943, 'learning_rate': 9.999999999999999e-06, 'epoch': 0.0}                                       
{'loss': 0.6841, 'grad_norm': 2.78806209564209, 'learning_rate': 1.1999999999999999e-05, 'epoch': 0.0}                                        
{'loss': 1.162, 'grad_norm': 3.4792563915252686, 'learning_rate': 1.4e-05, 'epoch': 0.0}                                                      
{'loss': 0.6002, 'grad_norm': 2.4968347549438477, 'learning_rate': 1.6e-05, 'epoch': 0.0}                                                     
{'loss': 0.7202, 'grad_norm': 2.28143310546875, 'learning_rate': 1.7999999999999997e-05, 'epoch': 0.01}                                       
{'loss': 0.661, 'grad_norm': 2.548154592514038, 'learning_rate': 1.9999999999999998e-05, 'epoch': 0.01}                                       
{'loss': 0.6542, 'grad_norm': 2.4695241451263428, 'learning_rate': 2.2e-05, 'epoch': 0.01}                                                    
{'loss': 0.6747, 'grad_norm': 3.059981346130371, 'learning_rate': 2.3999999999999997e-05, 'epoch': 0.01}                                      
{'loss': 0.5796, 'grad_norm': 1.9567434787750244, 'learning_rate': 2.6e-05, 'epoch': 0.01}                                                    
{'loss': 0.7219, 'grad_norm': 2.179964780807495, 'learning_rate': 2.8e-05, 'epoch': 0.01}                                                     
{'loss': 0.565, 'grad_norm': 1.6430941820144653, 'learning_rate': 2.9999999999999997e-05, 'epoch': 0.01}                                      
{'loss': 0.85, 'grad_norm': 1.84047532081604, 'learning_rate': 3.2e-05, 'epoch': 0.01}                                                        
{'loss': 0.3432, 'grad_norm': 1.1440203189849854, 'learning_rate': 3.399999999999999e-05, 'epoch': 0.01}                                      
{'loss': 0.2684, 'grad_norm': 1.2846965789794922, 'learning_rate': 3.5999999999999994e-05, 'epoch': 0.01}                                     
{'loss': 0.5078, 'grad_norm': 4.036565780639648, 'learning_rate': 3.8e-05, 'epoch': 0.01}                                                     
{'loss': 0.5078, 'grad_norm': 1.8525359630584717, 'learning_rate': 3.9999999999999996e-05, 'epoch': 0.01}                                     
{'loss': 0.3205, 'grad_norm': 2.1137163639068604, 'learning_rate': 4.2e-05, 'epoch': 0.01}                                                    
{'loss': 0.2558, 'grad_norm': 1.8392887115478516, 'learning_rate': 4.4e-05, 'epoch': 0.01}                                                    
{'loss': 0.5443, 'grad_norm': 1.7691305875778198, 'learning_rate': 4.599999999999999e-05, 'epoch': 0.01}                                      
{'loss': 0.4253, 'grad_norm': 3.6443562507629395, 'learning_rate': 4.7999999999999994e-05, 'epoch': 0.01}                                     
{'loss': 0.3279, 'grad_norm': 1.0808053016662598, 'learning_rate': 4.9999999999999996e-05, 'epoch': 0.02}                                     
{'loss': 0.43, 'grad_norm': 1.9154982566833496, 'learning_rate': 5.2e-05, 'epoch': 0.02}                                                      
{'loss': 0.446, 'grad_norm': 1.612378716468811, 'learning_rate': 5.399999999999999e-05, 'epoch': 0.02}                                        
{'loss': 0.3042, 'grad_norm': 1.5698323249816895, 'learning_rate': 5.6e-05, 'epoch': 0.02}                                                    
{'loss': 0.2976, 'grad_norm': 1.4310812950134277, 'learning_rate': 5.7999999999999994e-05, 'epoch': 0.02}                                     
{'loss': 0.5587, 'grad_norm': 1.7023086547851562, 'learning_rate': 5.9999999999999995e-05, 'epoch': 0.02}                                     
{'loss': 0.3389, 'grad_norm': 1.5704988241195679, 'learning_rate': 6.199999999999999e-05, 'epoch': 0.02}                                      
{'loss': 0.3794, 'grad_norm': 1.9065048694610596, 'learning_rate': 6.4e-05, 'epoch': 0.02}                                                    
{'loss': 0.1972, 'grad_norm': 1.6585965156555176, 'learning_rate': 6.599999999999999e-05, 'epoch': 0.02}                                      
{'loss': 0.3562, 'grad_norm': 1.0364704132080078, 'learning_rate': 6.799999999999999e-05, 'epoch': 0.02}                                      
{'loss': 0.3585, 'grad_norm': 1.6444295644760132, 'learning_rate': 7e-05, 'epoch': 0.02} 
```
</details>

<br>

>**Question2**
>Following the guidelines from Question 1, repeat those experiments for FFT training. Observe the time taken to train and GPU memory usage. Then, with the optimal training setting, compare LoRA and FFT in terms of these two metrics. Analyze and explain the reasons behind any differences observed between the two training methods.

Deleting --load-8-bit will remove compression and increase GPU memory 

```bash
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 79.26 GiB of which 60.75 MiB is free. Including non-PyTorch memory, this process has 79.19 GiB memory in use.
```
Using --load-4-bit reduces GPU memory usuage

```bash
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000001:00:00.0 Off |                    0 |
| N/A   62C    P0            304W /  300W |   31521MiB /  81920MiB |    100%      Default |
|                                         |                        |             Disabled |
```

Therefore, GPU memory usage: 16bit>8bit>4bit

>**Question3**
>Inference

<br>
CUDA_VISIBLE_DEVICES=0 python inference.py --model_name_or_path /public/Meta-Llama-3-8B-Instruct-Lora --sid 100
<details>

```bash
>>>>>>>>>>>>>>>>>>>>>>>Prompt:
Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate answer to the given question.
### Instruction

Given a table and a list of texts in the following, what is the answer to the question? 

Please predict the answer and store it in a variable named `{answer}`. If there are multiple values, separate them using the '#' symbol.

If the value of the `{answer}` is numerical, predict its scale and store it in a variable named `{scale}`. 

The value of `{scale}` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `{scale}` to 'none'.



Finally, present the final answer in the format of "The answer is: {answer} #### and its corresponding scale is: {scale}"



### Table 

| | |Fiscal | |
| |2019 |2018 |2017 |
| | |(in millions) | |
|Transportation Solutions: | | | |
|Automotive |$ 5,686 |$ 6,092 |$  5,228 |
|Commercial transportation |1,221 |1,280 |997 |
|Sensors |914 |918 |814 |
|Total Transportation Solutions |7,821 |8,290 |7,039 |
|Industrial Solutions: | | | |
|Industrial equipment |1,949 |1,987 |1,747 |
|Aerospace, defense, oil, and gas |1,306 |1,157 |1,075 |
|Energy |699 |712 |685 |
|Total Industrial Solutions |3,954 |3,856 |3,507 |
|Communications Solutions: | | | |
|Data and devices |993 |1,068 |963 |
|Appliances |680 |774 |676 |
|Total Communications Solutions |1,673 |1,842 |1,639 |
|Total |$ 13,448 |$ 13,988 |$ 12,185 |




### Text

1 Net sales by segment and industry end market(1) were as follows:
2 (1) Industry end market information is presented consistently with our internal management reporting and may be revised periodically as management deems necessary.




### Question 

What was the change in the amount for Appliances in 2019 from 2018?



### Response


<<<<<<<<<<<<<<<<<<<<<<<Model Response:
The answer is: -94 #### and its corresponding scale is: million


<<<<<<<<<<<<<<<<<<<<<<<Sample Answer:
The answer is: -94 #### and its corresponding scale is: million

```

</details>
<br>
CUDA_VISIBLE_DEVICES=0 python inference.py --model_name_or_path /public/Meta-Llama-3-8B-Instruct-FFT --sid 100
<details>

```bash
>>>>>>>>>>>>>>>>>>>>>>>Prompt:
Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate answer to the given question.



### Instruction

Given a table and a list of texts in the following, what is the answer to the question? 

Please predict the answer and store it in a variable named `{answer}`. If there are multiple values, separate them using the '#' symbol.

If the value of the `{answer}` is numerical, predict its scale and store it in a variable named `{scale}`. 

The value of `{scale}` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `{scale}` to 'none'.



Finally, present the final answer in the format of "The answer is: {answer} #### and its corresponding scale is: {scale}"



### Table 

| | |Fiscal | |
| |2019 |2018 |2017 |
| | |(in millions) | |
|Transportation Solutions: | | | |
|Automotive |$ 5,686 |$ 6,092 |$  5,228 |
|Commercial transportation |1,221 |1,280 |997 |
|Sensors |914 |918 |814 |
|Total Transportation Solutions |7,821 |8,290 |7,039 |
|Industrial Solutions: | | | |
|Industrial equipment |1,949 |1,987 |1,747 |
|Aerospace, defense, oil, and gas |1,306 |1,157 |1,075 |
|Energy |699 |712 |685 |
|Total Industrial Solutions |3,954 |3,856 |3,507 |
|Communications Solutions: | | | |
|Data and devices |993 |1,068 |963 |
|Appliances |680 |774 |676 |
|Total Communications Solutions |1,673 |1,842 |1,639 |
|Total |$ 13,448 |$ 13,988 |$ 12,185 |




### Text

1 Net sales by segment and industry end market(1) were as follows:
2 (1) Industry end market information is presented consistently with our internal management reporting and may be revised periodically as management deems necessary.




### Question 

What was the change in the amount for Appliances in 2019 from 2018?



### Response


<<<<<<<<<<<<<<<<<<<<<<<Model Response:
The answer is: -94 #### and its corresponding scale is: million


<<<<<<<<<<<<<<<<<<<<<<<Sample Answer:
The answer is: -94 #### and its corresponding scale is: million

```
</details>

<br>

>**Question4**
<br>
>Run inference on the original Llama2 and Llama3 models without fine-tuning. Specifying the model name of `/public/Meta-Llama-3-8B-Instruct` or `/public/` in the inference instruction: `CUDA_VISIBLE_DEVICES=0 python inference.py [--model_name_or_path MODEL_NAME_OR_PATH] --sid 100`. Evaluate the outputs from both Llama2 and Llama3 by comparing the model responses to the sample answers. Vary the sid 100 parameter to test different IDs like 100, 500, 900, among others, to review their performance. Based on your analysis, which model do you think performs better?

Part4.1 Sid=100

```bash
| |2019 |2018 |2017 |
| | |(in millions) | |
|Appliances |680 |774 |676 |
What was the change in the amount for Appliances in 2019 from 2018?
```

Llama3: 
```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
### Answer

The answer is: $94 #### and its corresponding scale is: none

### Explanation

To find the change in the amount for Appliances in 2019 from 2018, we need to subtract the value in 2018 from the value in 2019. According to the table, the value for Appliances in 2018 is $774 million and in 2019 is $680 million. The change is:

$774 million - $680 million = $94 million

Since the value is in millions, the scale is 'none'.<|eot_id|>
```

Llama2:
```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
 The answer to the question is:

The change in the amount for Appliances in 2019 from 2018 is: $774 - $774 = $0.


The scale of the answer is: none.


The final answer is: The change in the amount for Appliances in 2019 from 2018 is $0 #### and its corresponding scale is none.
```

Comparing the 2 models with sid 100, Llama3 extracts the correct data and computes the answers correctly. 

Part4.2 Sid=900
```bash
| |As of July 31, | |
| |2019 |2018 |
|Less valuation allowance |31,421 |28,541 |

What was the valuation allowance in 2019 and 2018 respectively?
```

Llama3:

```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
### Answer

The answer is: 31,421#28,541

### Scale

The scale is: none<|eot_id|>
```
Llama2:

```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
 The answer to the question is:

The valuation allowance in 2019 was $31,421,000 and in 2018 it was $28,541,000.

The scale of the valuation allowance is 'million'.
```

Surprisingly for sid=900, Llama2 was able to get the correct scale and numbers right. 


>**Question5**
<br>
>Run inference on the models that have been fine-tuned and compare the difference between model answer and sample answer. Again, you can test different sample IDs by adjusting the `sid 100` parameter to observe various results. Which model do you think performs the best: the original, LoRA fine-tuned, or FFT fine-tuned?

Meta-Llama-3-8B-Instruct-Lora

```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
The answer is: $31.4 million#$28.5 million #### and its corresponding scale is: none
```

Meta-Llama-3-8B-Instruct-FFT

```bash
<<<<<<<<<<<<<<<<<<<<<<<Model Response:
The answer is: $31.4 million#$28.5 million #### and its corresponding scale is: none
```