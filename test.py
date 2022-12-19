import paddle
import data
from utils.build_ref import build_ref_trainer

trainer = build_ref_trainer(751, batch_size=16)
for x in trainer.data_loader:
    print()
print()