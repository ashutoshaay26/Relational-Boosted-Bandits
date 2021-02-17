# Util file for creating batch data given dataset,.
import os
import random
import shutil
import subprocess
# Count total lines.
def count_lines(filepath):
	lines = sum(1 for line in open(filepath))
	return lines

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



# Create Train Files
def create_train_batch_files(batch_size=50):
	


	batch_filepath = "Meta_data/train/"
	
	'''
	_CALL = ("rm Meta_data/train/*")
	_call_shell_command(_CALL)
	'''

	##########		
	# Train_pos_ Generation
	##########

	#lines = count_lines(src_filepath)
	with open("train_pos.txt") as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	lines = len(content)
	random.shuffle(content)

	if batch_size*2>= lines:
		batch_size = 1

	total_batches = lines // batch_size

	# Writing batch into files.
	for i in range(total_batches-1):
		temp = content[i*batch_size : (i*batch_size+ batch_size) ]
		with open(batch_filepath+'train_pos_'+str(i)+'.txt', 'w') as a_writer:
			for j in temp:		
				a_writer.write(j+'\n')

	# Wrinting last batch into file.
	with open(batch_filepath+'train_pos_'+str(total_batches-1)+'.txt', 'w') as a_writer:
		for j in temp[total_batches-1:]:		
			a_writer.write(j+'\n')

	##########		
	# Train_neg_ Generation
	##########
	#lines = count_lines(src_filepath)
	with open("train_neg.txt") as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	lines = len(content)
	random.shuffle(content)

	if batch_size*2>= lines:
		batch_size = 1

	batch_size = lines // total_batches

	# Writing batch into files.
	for i in range(total_batches-1):
		temp = content[i*batch_size : (i*batch_size+ batch_size) ]
		with open(batch_filepath+'train_neg_'+str(i)+'.txt', 'w') as a_writer:
			for j in temp:		
				a_writer.write(j+'\n')

	# Wrinting last batch into file.
	with open(batch_filepath+'train_neg_'+str(total_batches-1)+'.txt', 'w') as a_writer:
		for j in temp[total_batches-1:]:		
			a_writer.write(j+'\n')




	print(total_batches,lines)

'''
# Create Test Files
def create_test_batch_files():
	batch_filepath = "Meta_data/test/"

	lines = count_lines(src_filepath)
	#lines = count_lines(src_filepath)
	with open(filename) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	lines = len(content)
	random.shuffle(content)

'''

def create_batch_files(batch_size=50):
	create_train_batch_files(batch_size)
	#create_test_batch_files()

if __name__ == '__main__':
	batch_size = 200
	create_batch_files(batch_size)