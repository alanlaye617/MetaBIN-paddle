import sys
sys.path.append('.')
from utils import build_ref_trainer
from utils import read_image_ref
from data.datasets.utils import read_image as read_image_pad
from data import build_reid_test_loader
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper

reprod_log_ref = ReprodLogger()
reprod_log_pad = ReprodLogger()  
batch_size = 16

trainer_ref = build_ref_trainer(batch_size=batch_size, resume=True)
dataloader_ref, num_query = trainer_ref.build_test_loader(trainer_ref.cfg, 'DukeMTMC')
dataloader_pad, num_query= build_reid_test_loader('DukeMTMC', batch_size=batch_size, num_workers=2, flag_test=True)

transform_pad = dataloader_pad.dataset.transform
transform_ref = dataloader_ref.dataset.transform

path = './1.jpg'
img_pad = read_image_pad(path)
img_ref = read_image_ref(path)

reprod_log_pad.add("original",  np.asarray(img_pad))
reprod_log_ref.add("original",  np.asarray(img_ref))

transformed_img_pad = transform_pad(img_pad)
reprod_log_pad.add("transformed",  transform_pad(img_pad).cpu().detach().numpy())

transformed_img_ref = transform_ref(img_ref)
reprod_log_ref.add("transformed",  transform_ref(img_ref).cpu().detach().numpy())

reprod_log_ref.save('./result/transforms_ref.npy')
reprod_log_pad.save('./result/transforms_paddle.npy')


diff_helper = ReprodDiffHelper()
info1 = diff_helper.load_info("./result/transforms_paddle.npy")
info2 = diff_helper.load_info("./result/transforms_ref.npy")
diff_helper.compare_info(info1, info2)

diff_helper.report(
    diff_method="mean", diff_threshold=1e-6, path="./result/log/transforms_diff.log")
    
print()