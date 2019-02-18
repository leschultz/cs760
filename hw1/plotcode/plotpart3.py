from matplotlib import pyplot as pl


def parser(name):
    '''
    Parse the output files for part 3

    inputs:
        name = The name of the file containing data

    outputs:
    '''

    with open(name) as file:
        for line in file:
            values = line.strip().split(',')

            if len(values) > 1:
                k['votes'].append(int(values[0]))
                accuracies['votes'].append(float(values[-1]))


pathvotes = '../plotdata/part3votes'
pathdigits = '../plotdata/part3digits'

pathsvotes = []
pathsdigits = []
for i in range(2, 10+1, 2):
    pathsvotes.append(pathvotes+str(i))
    pathsdigits.append(pathdigits+str(i))

percents = [i for i in range(10, 100+1, 10)]

k = {
     'votes': [],
     'digits': []
     }

accuracies = {
              'votes': [],
              'digits': []
              }

with open(pathvotes) as file:
    for line in file:
        values = line.strip().split(',')

        if len(values) > 1:
            k['votes'].append(int(values[0]))
            accuracies['votes'].append(float(values[-1]))

with open(pathdigits) as file:
    for line in file:
        values = line.strip().split(',')
        
        if len(values) > 1:
            k['digits'].append(int(values[0]))
            accuracies['digits'].append(float(values[-1]))

fig, (axvotes, axdigits) = pl.subplots(2, 1)

axvotes.plot(k['votes'], accuracies['votes'], marker='.', label='Votes', color='b')
axvotes.set_xlabel('k-nearest neighbors')
axvotes.set_ylabel('Accuracy')
axvotes.grid()
axvotes.legend(loc='best')

axdigits.plot(k['digits'], accuracies['digits'], marker='.', label='Digits', color='r')
axdigits.set_xlabel('k-nearest neighbors')
axdigits.set_ylabel('Accuracy')
axdigits.grid()
axdigits.legend(loc='best')

fig.tight_layout()
pl.savefig('../plotimages/tune_k.pdf')
