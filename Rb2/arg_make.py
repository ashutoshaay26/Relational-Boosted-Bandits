






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

