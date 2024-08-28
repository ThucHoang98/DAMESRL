DAMESRL is a framework for deep semantic role labeling, which doesn't require feature engineering.

1. System requirement:
Python 3
Tensorflow-gpu (for gpu) or tensorflow (for cpu)

2. Input format:
Our input format is a shortened version of the CoNLL'05 format, which only contains the Words, Targets and (possibly) Props columns (See http://www.lsi.upc.edu/\~srlconll/conll05st-release/README}. The framework also provides scripts to convert the other formats such as the full CoNLL'05, CoNLL'09 and CoNLL'12 automatically to our format:
- python liir/dame/core/io/CoNLL2005Reader.py  path_to_conll2005_files  output
- python liir/dame/core/io/CoNLL2012Reader.py  path_to_conll2012_files  output

3. Train a new model:
- Edit the config file, note that "mode" must be "train". We provide some samples of the config files corresponding to the models reported in the paper.

python liir/dame/srl/DSRL.py path_to_config_file GPU_ID


4. Predict
We can you the same config file which was used to train the model. 
- Change "mode" to "infer" and "tag" to "best" or any other checkpoint. 
- Specify the path to the input: test: data/srl/samples/test-brown
python liir/dame/srl/DSRL.py path_to_config_file GPU_ID

5. Visualize the output

python liir/dame/core/io/HTMLWriter.py path_to_input/output_file path_to_output_html_file


7. Scripts

We can perform training and evaluating using scripts/run.sh:
- Go to the home folder of DAMESRL
- sh scripts/run.sh config_file gpu_id

Note that you need to modify the configuration file before running the script.

Format converter:
- sh scripts/convert.sh type input output
type= 05 for CoNLL2005 or 12 for CoNLL2012 

Visualizing the data:
- sh scripts/visual.sh input output (the output should be an html file)

