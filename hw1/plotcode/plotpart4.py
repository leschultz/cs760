from matplotlib import pyplot as pl


def parser(name):
    '''
    Parse the output files for part 3

    inputs:
        name = The name of the file containing data

    outputs:
    '''

    fpr = []
    tpr = []
    with open(name) as file:
        for line in file:
            values = line.strip().split(',')

            if len(values) > 1:
                fpr.append(float(values[0]))
                tpr.append(float(values[-1]))

    return fpr, tpr


pathvotes = '../plotdata/part4roc'

pathsvotes = []
pathsdigits = []
kvals = []
for i in [10, 20, 30]:
    pathsvotes.append(pathvotes+str(i))
    kvals.append(i)

percents = [i for i in range(10, 100+1, 10)]

voteslengths = []
votesacc = []
for name in pathsvotes:
    items = parser(name)
    voteslengths.append(items[0])
    votesacc.append(items[1])
    
fig, axvotes = pl.subplots()

count = 0
for k in kvals:
    axvotes.plot(voteslengths[count], votesacc[count], marker='.', label='k='+str(k))
    count += 1

axvotes.set_xlabel('FPR for votes data')
axvotes.set_ylabel('TPR for votes data')
axvotes.grid()
axvotes.legend(loc='lower right')

fig.tight_layout()
pl.savefig('../plotimages/roc_curve.pdf')
