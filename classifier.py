import csv as csv
import numpy as np

import neuralnetwork

csv_file_object = csv.reader(open('train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data=[] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 	# adding each row to the data variable
idx = 2*len(data)/3
print idx
data = np.array(data).astype(np.float)								# Then convert from a list to an array.



training_data = data[0:idx]
y = training_data[0::,0]
y_oneshot = []
for digit in y:
    r =  [0.0]*10
    r[int(digit)] = 1.0
    y_oneshot.append(r)

y_oneshot = np.array(y_oneshot)
y_oneshot = [np.reshape(y, (10,1)) for y in y_oneshot ]
x_input = [np.reshape(x, (784, 1)) for x in training_data[0::,1::]]


training_data = zip(x_input ,y_oneshot)

test_data = data[idx:]
x_input = [np.reshape(x, (784, 1)) for x in test_data[0::,1::]]
test_data = zip(x_input, test_data[0::,0])



net = neuralnetwork.Network([784, 100, 10])
net.SGD(training_data, 50, 10, 0.01)
#train

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

predictions_file = open("submission.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["ImageId", "Label"])	# write the column headers
i=1
for row in test_file_object:
    row = np.reshape(np.array(row[0:]).astype(np.float), (784, 1))
    test_results = np.argmax(net.feedforward(row))
    predictions_file_object.writerow([i, int(test_results)])
    i = i+1
test_file.close()												# Close out the files.
predictions_file.close()
