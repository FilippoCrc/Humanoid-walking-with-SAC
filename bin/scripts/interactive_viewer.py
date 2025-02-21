import os

import mujoco
from mujoco import viewer
from spqr_rl_mujoco.utils.misc import project_root

os.environ["MUJOCO_GL"] = "egl"
root = project_root()

model = mujoco.MjModel.from_xml_path(f"{root}/spqr_rl_mujoco/model/scene.xml")
data = mujoco.MjData(model)

# mujoco.mj_resetDataKeyframe(model, data, 2)

viewer.launch(model, data)
