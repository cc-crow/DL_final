import mmdet
from mmdet.apis import init_detector, inference_detector
from mmdet.models.detectors import BaseDetector
import mmcv
import cv2

config_path = '1/'
model_path = '1/'
config_file = config_path+'imagenet_pre_config.py'
checkpoint_file = model_path+'epoch6.pth'

# Loading the model from the config and saved checkpoint, inference to GPU
model = init_detector(config_file, checkpoint_file, device='cuda:3')

# test a single image and show the results
for i in ['']:
    img = 'test{}.jpg'.format(i)  # or img = mmcv.imread(img), which will only load it once
    #img = mmcv.imread(img)
    result = inference_detector(model, img)
    # visualize the results in a new window
    #show_result(img, result, model.CLASSES, show=False)
    # or save the visualization results to image files
    #BaseDetector.show_result(img, result, model.CLASSES, show=False, out_file='result.jpg')
    #cv2.imwrite('result.jpg', result)

    from mmdet.apis import show_result_pyplot
    show_result_pyplot(model, img, result, out_file='{}-result-{}.jpg'.format(config_file.replace(config_path, "").split('_')[0], i))