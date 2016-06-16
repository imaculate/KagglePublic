import csv as csv
import numpy as np


csv_file_object = csv.reader(open('train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next()
data=[] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:])
data = np.array(data).astype(np.float32)


m = len(data)
y = data[0::,0]
X = data[0::,1::]


y_oneshot = []
for digit in y:
    r =  [0.0]*10
    r[int(digit)] = 1.0
    y_oneshot.append(r)
t = np.array(y_oneshot)
y_oneshot = []



d = 784
#lambas = np.logspace(1, -6, num=8)
lambd = 0.001

#nhs = np.linspace(100, 1500,num = 15)
nh = 600

Xa = np.append(np.ones((m,1)), X, axis=1)                   # append ones to X (bias term)


Wh = np.random.rand(d+1,nh)                    # hidden weights from U(0,1)
Wh[1:,:]= 2*Wh[1:,:]-1          # remap nonbias weights to -1,+1
H = 1./(1+np.exp(-1*np.matmul(Xa,Wh)))                # design matrix
    #Hi = np.linalg.pinv(H);                          # calculate the pseudoinverse of the design matrix
temp = np.linalg.pinv(np.matmul(np.matrix.transpose(H),H) +(lambd*np.eye(nh)))
Hi=np.matmul(temp,np.matrix.transpose(H)) # regularised pseudoinverse

w = np.matmul(Hi,t)                             # fitted linear weights
Hi = []

# tpred = np.matmul(H, w)
# HitAccuracy = 0;
# #
# for i in range(m):
#     label_index_actual= np.argmax(tpred[i])
#     label_index_expected= np.argmax(t[i])
#     if label_index_actual==label_index_expected:
#         HitAccuracy = HitAccuracy + 1
#     #print HitAccuracy
# percent = (HitAccuracy*100/m)
# print percent, 'percent were classified correctly'

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
#
test = []
for row in test_file_object: 							# Skip through each row in the csv file,
    test.append(row[0:])
m = len(test)
test = np.array(test).astype(np.float32)

testa = np.append(np.ones((m,1)), test, axis=1)
Ha = 1./(1+np.exp(-1*np.matmul(testa,Wh)))
tq = np.matmul(Ha, w)
Ha = []

predictions_file = open("submission2.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["ImageId", "Label"])	# write the column headers

for i in range(m):
    test_results = np.argmax(tq[i])
    predictions_file_object.writerow([i+1, test_results]  )

test_file.close()												# Close out the files.
predictions_file.close()


