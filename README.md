# Description
run inference to use pretrained model to inference on new senenteces. 
Parameters to the inference files should be the config path, checkpoint path (should be working_dir) and path to text file. The text file should contain in each line
sentence, source rate, target rate. The output will be printed.
Source and target should be a number between 1 to 5 where 1 is the negative and 5 is the most positive.
