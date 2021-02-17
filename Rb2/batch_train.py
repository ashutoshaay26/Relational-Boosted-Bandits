import os
import shutil
import subprocess
import numpy as np
import param

from exploration import *

from time import sleep
def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])
# Read File 
def read_file(filename):
	with open(filename) as f:
		content = f.readlines()
	return(content)	

class online_boosting():
    def __init__(self, target = "None",n_trees = 5):
        self.target = target # The predicate we want to predict.
        self.n_trees = n_trees # number of trees to learn.
        #self.n_batches = n_batches # Total Batches to learn on models.


    def _call_shell_command(self,shell_command):
        """Start a new process to execute a shell command.
        This is intended for use in calling jar files. It opens a new process and
        waits for it to return 0.
        Parameters
        ----------
        shell_command : str
            A string representing a shell command.
        Returns
        -------
        None
        """

        _pid = subprocess.Popen(shell_command, shell=True)
        _status = _pid.wait()
        if _status != 0:
            raise RuntimeError(
                "Error when running shell command: {0}".format(shell_command)
            )

    # Train the model
    def train_call(self, total_trees):
        _CALL = (
            "java -jar "
            + str(param.BOOST_JAR)
            + " -l -train "
            + str(param.TRAIN_DIR)
            + " -target "
            + str(self.target)
            + " -trees "
            + str(total_trees + self.n_trees)
            + " -NegPosRatio "
            + str(1)
        )

        # Call the constructed command.
        self._call_shell_command(_CALL)
        print("Completed the training..")


    # Infer from model
    def inference_call(self):
        total_trees = count_files(param.MODELS_DIR+"bRDNs/Trees/")
        _CALL = (
                "java -jar "
                + str(param.BOOST_JAR)
                + " -i -test "
                + str(param.TEST_DIR)
                + " -model "
                + str(param.MODELS_DIR)
                + " -target "
                + self.target
                + " -trees "
                + str(total_trees)
            )

        self._call_shell_command(_CALL)
        print("Completed the Prediction Task")

    def batch_training(self):
        total_trees = 0
        model_path = param.MODELS_DIR+"bRDNs/"
        #self.train(total_trees)

        train_src = "Meta_data/train/" 
        test_src  = "Meta_data/test/"
        
        train_dst = "train/"
        test_dst  = "test/"

    ##### Cold Start Here, First batch train using random policy! 
        
        shutil.copy2(train_src+"train_pos_"+str(0)+".txt", train_dst+"train_pos.txt")
        shutil.copy2(train_src+"train_neg_"+str(0)+".txt", train_dst+"train_neg.txt")
        #total_trees = count_files(param.MODELS_DIR+"bRDNs/Trees/")
        self.train_call(total_trees)
        
    ##### Cold Start ends here..............

        print("Cold Start Completed....")
        sleep(1)
        bulk_data = read_file(train_src + 'train_pos_bulk.txt')
        print("Interrupt",param.total_samples)
        cnt_sample=0
        while cnt_sample <param.total_samples: 
            # Replace the train_data and train_test from respective folder.

            cnt_sample = gen_explore_train(bulk_data,param.batch_size,cnt_sample)
            '''
            Renaming the model file here for warm start.
            '''
            if os.path.exists(model_path+self.target+".model"):
                os.rename(model_path+self.target+".model",model_path+self.target+".model"+".ckpt")
                total_trees = count_files(param.MODELS_DIR+"bRDNs/Trees/")
                print("Yessssssss : ", total_trees)
    
            if (sum(1 for line in open(train_dst+"train_pos.txt")) != 0):
                self.train_call(total_trees)
            else:
                os.rename(model_path+self.target+".model"+".ckpt",model_path+self.target+".model")


if __name__ == '__main__':
    target = "like"
    n_trees = 2
    #n_batches = count_files("Meta_data/train/") //2
    boost = online_boosting(target,n_trees)

    # Learn the Models
    #boost.train()
    boost.batch_training()
    #predict from models
    #boost.inference()

    #print(boost.predict_prob())
    
    #print(count_files(param.MODELS_DIR+"bRDNs/Trees/"))