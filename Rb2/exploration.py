import os
import shutil
import subprocess
import numpy as np
import param

import pickle
import random
import math

from time import sleep

rd_label={0:'horror_0',1:'horror_1',2:'action_0',3:'action_1',4:'drama_0',5:'drama_1'} # Reverse Dictionary
d_label={'horror_0':0,'horror_1':1,'action_0':2,'action_1':3,'drama_0':4,'drama_1':5} # Order Dictionary
counts = [0 for col in range(param.actions)]
values = [0.0 for col in range(param.actions)]
temperature = 0.1

# Write into pickle file.
def write_pickle(filename,file):
	with open(filename, 'wb') as handle:
		pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
	time.sleep(3)
	


# Function to read pickle file and read respected object.
def read_pickle(picklename):
	with open(picklename, 'rb') as handle:
		a = pickle.load(handle)
	return a		
def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])


# Break the test query into target,
def break_query(query):
	first_break = query.find('(')
	second_break = query.find(',')
	third_break = query.find(')')
	target = query[:first_break].strip()
	data_p = query[first_break+1 : second_break].strip()
	label = query[second_break+1: third_break].strip()  
	return (target,data_p,label)


def predict_prob(target):
    """Return class probabilities.

    Reading probabilites from results_(self.target).db file
    --------
    Returns
    -------
    results : List of tuples. 
        Query and Probability of belonging to the positive class of test examples. 
    """

    _results_db ="test/"+"results_" + target + ".db"
    _classes, _results = np.loadtxt(
        _results_db,
        delimiter=")",
        usecols=(0, 1),
        converters={0: lambda s: 0 if s[0] == 33 else 1},
        unpack=True,
    )

    _neg = _results[_classes == 0]
    _pos = _results[_classes == 1]

    _results2 = np.concatenate((_pos, 1 - _neg), axis=0)

    _query_db = "test/"+"query_" + target + ".db"
    queries = open(_query_db, 'r').readlines()

    
    queries = list(map(lambda x: (x[1:].strip('\n') if ord(x[0])==33 else x.strip('\n')) , queries  ) ) # Data type is changed so haveto write ord().
    #print(queries)
    #return _classes

    # Return array of tuples (query, prob)
    final_result = []
    for i,j in zip(queries,_results2):
        final_result.append((i,j))
    
    return queries,_results2

def _call_shell_command(shell_command):
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

def inference_call(target):
    total_trees = count_files(param.MODELS_DIR+"bRDNs/Trees/")
    print(total_trees)
    sleep(0.7)
    _CALL = (
            "java -jar "
            + str(param.BOOST_JAR)
            + " -i -test "
            + str(param.TEST_DIR)
            + " -model "
            + str(param.MODELS_DIR)
            + " -target "
            + target
            + " -trees "
            + str(total_trees)
        )

    _call_shell_command(_CALL)
    print("Inference Call Done...")



def sample_draw(probs):
	z = random.random()
	cum_prob = 0.0
	for i in range(len(probs)):
		prob = probs[i]
		cum_prob += prob
		if cum_prob > z:
			return i
	return len(probs) - 1


def select_pos_sample(pos,batchsize):
	pro = []
	samples=[]
	final = []
	for p,s in pos:
		pro.append(1-p)
		samples.append(s)
	if len(pos)<=batchsize:
		return samples
	for i in range(batchsize):	
		z = sum([math.exp(v) for v in pro])
		probs = [math.exp(v) / z for v in pro]
		index = sample_draw(probs)
		print(index)
		final.append(samples.pop(index))
		pro.pop(index)
	return final	


def select_neg_sample(neg,batchsize):
	pro = []
	samples=[]
	final = []
	for p,s in neg:
		pro.append(p)
		samples.append(s)
	if len(neg)<=batchsize:
		return samples
	for i in range(batchsize):	
		z = sum([math.exp(v) for v in pro])
		probs = [math.exp(v) / z for v in pro]
		index = sample_draw(probs)
		final.append(samples.pop(index))
		probs.pop(index)

	return final

# Softmax over all actions, and then sample according to probabilities	
def categorical_draw(probs):
	z = random.random()
	cum_prob = 0.0
	for i in range(len(probs)):
		prob = probs[i]
		cum_prob += prob
		if cum_prob > z:
			return i
	return len(probs) - 1


def select_arm(pro):
	z = sum([math.exp(v / temperature) for v in pro])
	probs = [math.exp(v / temperature) / z for v in pro]
	return categorical_draw(probs)

def update(chosen_arm, reward):
	counts[chosen_arm] = counts[chosen_arm] + 1
	n = counts[chosen_arm]
	value = values[chosen_arm]
	new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
	values[chosen_arm] = new_value
	#return


def explor_softmax(target):
	queries,prob =  predict_prob(target)
	#print(queries)
	arm_number = select_arm(list(prob))

	return queries[arm_number],prob[arm_number]

def exploit(target):
	# Call Exploration Function Here.
	
	#sample = explor_softmax(target)
	sample,prob = explor_softmax(target)
	return sample,prob


# Generate Test file for one data point with all possible target. 
def gen_test_file(datap,target):
	target_file = "arg2.txt"
	test_dst  = "test/"

	with open(target_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]

	with open(test_dst+'test_pos.txt', 'w') as a_writer:
		for i in content:
			a_writer.write(target+'('+datap+','+i+').\n')

	# Empty test file.		
	with open(test_dst+'test_neg.txt', 'w') as a_writer:
		pass

######## Base Function ###################
def gen_explore_train(bulk_data,batchsize,cnt_sample):
	
	data_path = "Data/"

	#train_src = "Meta_data/train/" 
	#test_src  = "Meta_data/test/"

	train_dst = "train/"
	test_dst  = "test/"

	liked = read_pickle("like_movie.pkl")
	act_label = read_pickle("label.pkl")
	try:
		reward = np.load("reward.npy")
	except:
		reward = np.array([])

	success_query = []
	failed_query = [] 
	while(len(success_query)<batchsize or len(failed_query)<batchsize ):
	# with open(filepath) as f:
	# 	content = f.readlines()

	#content = [x.strip() for x in content]

	#for i in content:
		data_sample = bulk_data[random.randint(0,len(bulk_data)-1)]
		cnt_sample+=1
		target,datap,label = break_query(data_sample)

		gen_test_file(datap,target)
		
		inference_call(target)  # Inference for current Data point.

		suggest,prob = exploit(target) # Argmax over all possible labels.
		s_t,s_d,s_l = break_query(suggest)
		#cnt_ex[s_l]+=1        # Keep Tracking of Exploration for each label.
		if  len(liked[s_d.upper()])>0 and s_l in liked[s_d.upper()]:
			print(s_d)
			#print("YEYYY", label, s_l)
			reward=np.append(reward,1)
			success_query.append(suggest)
			#update(d_label[s_l],1)
		else:
			reward=np.append(reward,0)
			failed_query.append((prob,suggest))
			#update(d_label[s_l],0)
	#write_pickle('cnt_ex.pkl',cnt_ex)		

	# INformed Sampling.............
	success_query =  select_pos_sample(success_query)
	failed_query =  select_neg_sample(failed_query)

	##### Informed Sampling Block Ends here....
	
	#print("FQ:",failed_query,content)
	ch = 'w'
	with open(train_dst+'train_pos.txt', ch) as a_writer:
		for i in success_query:
			a_writer.write(i+'.'+'\n')

	with open(train_dst+'train_neg.txt', ch) as a_writer:
		for i in failed_query:
			a_writer.write(i[1]+'.'+'\n')

	np.save("reward.npy",reward)	
	return cnt_sample	

if __name__ == '__main__':

	gen_explore_train("Meta_data/train/train_neg_5.txt")
