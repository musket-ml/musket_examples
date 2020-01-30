import pandas as pd
import math

RESULT_CSV_RAW_PATH = "D:/kostya_work/runtime-New_configuration/TestProject/experiments/Bengali_0_01/predictions/validation0.[0]-pr.csv"
DST_PATH = "D:/kostya_work/runtime-New_configuration/TestProject/experiments/Bengali_0_01/predictions/validation0.[0]-pr-submission.csv"

srcDF = pd.read_csv(RESULT_CSV_RAW_PATH)
ids = srcDF['image_id']
l = len(ids)

items = []
rowIDs = set()
for rec in srcDF.items():
    cTitle = rec[0]
    if cTitle == 'image_id':
        continue
    series = rec[1]
    for i in range(l):
        imId = ids[i]
        rowId = f"{imId}_{cTitle}"
        val = series[i]

        if isinstance(val, float):
            if math.isnan(val):
                val = '0'
            else:
                val = str(int(val))
        elif isinstance(val, int):
            val = str(val)

        items.append({
            'row_id': rowId,
            'target': val
        })
        rowIDs.add(rowId)

prefLen = len("Train_")


def itemKey(x):
    s = x['row_id'][prefLen:]
    sep = s.index('_')
    ind = int(s[:sep])

    gr = s[sep + 1:]

    res = 0
    if gr == 'grapheme_root':
        res = 1
    elif gr == 'vowel_diacritic':
        res = 2
    return ind * 3 + res


sortedItems = sorted(items, key=itemKey)

submissionDF = pd.DataFrame(sortedItems, columns=['row_id', 'target'])

submissionDF.to_csv(DST_PATH,index=False)
