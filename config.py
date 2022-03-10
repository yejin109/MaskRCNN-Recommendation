# COCO
classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']

# Custom Dataset
# classes = (
#     'top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top',
#     'hood', 'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants',
#     'coat', 'jacket', 'jumper', 'padding jacket', 'best', 'kadigan',
#     'zip up', 'dress', 'jumpsuit')

batch_size = 1
score_threshold = 0.965
hidden_layer = 256
lr = 1e-3
max_size = 800
num_epochs = 5