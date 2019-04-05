from matplotlib import pyplot as pl

import pandas as pd

name_bag = 'bagged_tree_plot.pdf'
name_boost = 'boosted_tree_plot.pdf'

depth = list(range(1, 6))
dfbag = {}
dfboost = {}

dfbag['depth'] = depth

dfbag[3] = [
0.5524911032,
0.6254448399,
0.6112099644,
0.609430605,
0.6263345196,
]

dfbag[5] = [
0.7846975089,
0.8167259786,
0.8202846975,
0.8354092527,
0.8362989324,
]

dfbag[10] = [
0.859430605,
0.8443060498,
0.9021352313,
0.9101423488,
0.9190391459,
]

dfboost['depth'] = depth

dfboost[3] = [
0.5854092526690391,
0.43416370106761565,
0.5969750889679716,
0.7179715302491103,
0.7713523131672598,
]

dfboost[5] = [
0.7838078291814946,
0.7838078291814946,
0.8398576512455516,
0.8558718861209964,
0.8790035587188612,
]

dfboost[10] = [
0.8790035587,
0.8354092527,
0.9279359431,
0.9323843416,
0.9537366548,
]

dfbag = pd.DataFrame(dfbag)
dfboost = pd.DataFrame(dfboost)

# Bagging
figbag, axbag = pl.subplots()
axbag.plot(dfbag['depth'], dfbag[3], label='max depth: '+str(3))
axbag.plot(dfbag['depth'], dfbag[5], label='max depth: '+str(5))
axbag.plot(dfbag['depth'], dfbag[10], label='max depth: '+str(10))

axbag.set_xlabel('Number of Trees')
axbag.set_ylabel('Accuracy')
axbag.set_title('Bagging')
axbag.legend()
axbag.grid()

figbag.tight_layout()
figbag.savefig(name_bag)

# Boosting
figboost, axboost = pl.subplots()
axboost.plot(dfboost['depth'], dfboost[3], label='max depth: '+str(3))
axboost.plot(dfboost['depth'], dfboost[5], label='max depth: '+str(5))
axboost.plot(dfboost['depth'], dfboost[10], label='max depth: '+str(10))

axboost.set_xlabel('Maximum Number of Trees')
axboost.set_ylabel('Accuracy')
axboost.set_title('Boosting')
axboost.legend()
axboost.grid()

figboost.tight_layout()
figboost.savefig(name_boost)
