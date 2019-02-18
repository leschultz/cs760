from matplotlib import pyplot as pl


def parser(name):
    '''
    Parse the output files for part 3

    inputs:
        name = The name of the file containing data

    outputs:
    '''

    length = []
    accuracy = []
    with open(name) as file:
        for line in file:
            values = line.strip().split(',')

            if len(values) > 1:
                length.append(int(values[0]))
                accuracy.append(float(values[-1]))

    return length, accuracy


pathvotes = '../plotdata/part3votes'
pathdigits = '../plotdata/part3digits'

pathsvotes = []
pathsdigits = []
kvals = []
for i in range(2, 10+1, 2):
    pathsvotes.append(pathvotes+str(i))
    pathsdigits.append(pathdigits+str(i))
    kvals.append(i)

percents = [i for i in range(10, 100+1, 10)]

voteslengths = []
votesacc = []
for name in pathsvotes:
    items = parser(name)
    voteslengths.append(items[0])
    votesacc.append(items[1])

digitslengths = []
digitsacc = []
for name in pathsdigits:
    items = parser(name)
    digitslengths.append(items[0])
    digitsacc.append(items[1])
    
fig, (axvotes, axdigits) = pl.subplots(2, 1)

count = 0
for k in kvals:
    axvotes.plot(voteslengths[count], votesacc[count], marker='.', label='k='+str(k))
    count += 1

axvotes.set_xlabel('Traning Size')
axvotes.set_ylabel('Accuracy for Votes')
axvotes.grid()
axvotes.legend(loc='lower right')

count = 0
for k in kvals:
    axdigits.plot(digitslengths[count], digitsacc[count], marker='.', label='k='+str(k))
    count += 1

axdigits.set_xlabel('Training Size')
axdigits.set_ylabel('Accuracy for Digits')
axdigits.grid()
axdigits.legend(loc='lower right')

fig.tight_layout()
pl.savefig('../plotimages/learning_curve.pdf')
