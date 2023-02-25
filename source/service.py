import bentoml

import numpy as np
from bentoml.io import Image
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from json import JSONEncoder
import json

runner = bentoml.keras.get("bentoresnet50:2yjy44tr2k4zzqvp").to_runner()

svc = bentoml.Service("bentoresnet50", runners=[runner])

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@svc.api(input=Image(), output=JSON())
async def predict(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = await runner.async_run(arr)
    tag = ['normal', 'little', 'lot']
    predict = preds.tolist()
    numpyData = {"result": tag[(predict[0].index(max(predict[0])))]}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@' + tag[predict.index(max(predict))])
    print("1 list:", predict[0])
    print("2 max:", max(predict[0]))
    print("3 index:", predict[0].index(max(predict[0])))
    print("4 tag:", tag[(predict[0].index(max(predict[0])))])
    
    return encodedNumpyData