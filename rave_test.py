#%%
import torch

torch.set_grad_enabled(False)
from rave import RAVE
from prior import Prior

import librosa as li
import soundfile as sf

################ LOADING PRETRAINED MODELS ################
rave = RAVE.load_from_checkpoint("./models/amen3/amen3_rave.ckpt",
                                 strict=False).eval()
# prior = Prior.load_from_checkpoint("./models/amen2/amen2_prior.ckpt",
#                                    strict=False).eval()

#%%
################ RECONSTRUCTION ################

# STEP 1: LOAD INPUT AUDIO
x, sr = li.load("samples/voice_audio.wav", sr=rave.sr)

# STEP 2: ENCODE DECODE AUDIO
x = torch.from_numpy(x).reshape(1, 1, -1).float()
latent = rave.encode(x)
y = rave.decode(latent)

# STEP 3: EXPORT
sf.write("tmp/output_test_amen3.wav", y.reshape(-1).numpy(), sr)
# %%
