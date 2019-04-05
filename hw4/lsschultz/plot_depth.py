from matplotlib import pyplot as pl

import pandas as pd

name_bag = 'bagged_tree_plot.pdf'
name_boost = 'boosted_tree_plot.pdf'

depth = list(range(1, 7))
dfbag = {}
dfboost = {}

dfbag['depth'] = depth

dfbag[3] = [
0.1957295374,
0.3754448399,
0.6112099644,
0.7339857651,
0.8202846975,
0.8647686833,
]

dfbag[4] = [
0.1957295374,
0.3941281139,
0.609430605,
0.7393238434,
0.8354092527,
0.8772241993,
]

dfbag[5] = [
0.1930604982,
0.3887900356,
0.6263345196,
0.7366548043,
0.8362989324,
0.8861209964,
]

dfboost['depth'] = depth

dfboost[3] = [
0.1930604982,
0.3887900356,
0.6263345196,
0.7366548043,
0.8362989324,
0.8861209964,
]

dfboost[4] = [
0.1868327402,
0.4644128114,
0.7179715302,
0.8104982206,
0.8558718861,
0.9003558719,
]

dfboost[5] = [
0.159252669,
0.6423487544,
0.7713523132,
0.8425266904,
0.8790035587,
0.9074733096,
]

dfbag = pd.DataFrame(dfbag)
dfboost = pd.DataFrame(dfboost)

# Bagging
figbag, axbag = pl.subplots()
axbag.plot(dfbag['depth'], dfbag[3], label='#trees: '+str(3))
axbag.plot(dfbag['depth'], dfbag[4], label='#trees: '+str(4))
axbag.plot(dfbag['depth'], dfbag[5], label='#trees: '+str(5))

axbag.set_xlabel('Maximum Tree Depth')
axbag.set_ylabel('Accuracy')
axbag.set_title('Bagging')
axbag.legend()
axbag.grid()

figbag.tight_layout()
figbag.savefig(name_bag)

# Boosting
figboost, axboost = pl.subplots()
axboost.plot(dfboost['depth'], dfboost[3], label='max trees: '+str(3))
axboost.plot(dfboost['depth'], dfboost[4], label='max trees: '+str(4))
axboost.plot(dfboost['depth'], dfboost[5], label='max trees: '+str(5))

axboost.set_xlabel('Maximum Tree Depth')
axboost.set_ylabel('Accuracy')
axboost.set_title('Boosting')
axboost.legend()
axboost.grid()

figboost.tight_layout()
figboost.savefig(name_boost)
