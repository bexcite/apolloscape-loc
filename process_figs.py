# Utility script that combines and filtering figures images from a couple of runs
# used for GIF creation
import glob
import os
import shutil



# combine figures
fig_base_dir = '_checkpoints'
fig_dirs = ['20180821_141324_zpark_posenet_L1_resnet34p_2048',
           '20180822_072019_zpark_posenet_L1_resnet34p_2048',
           '20180822_173408_zpark_posenet_L1_resnet34p_2048',
           '20180823_085404_zpark_posenet_L1_resnet34p_2048']

result_dir = '__img_results_50'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def get_epoch(f):
    return int(f[f.rfind('_')+2:])


figs_all = {}
for fd in fig_dirs:
    fs = sorted(glob.glob(os.path.join(fig_base_dir, fd, '*.png')))
    figs_dict = {get_epoch(f[:-4]):f for f in fs}
    figs_all.update(figs_dict)
for i,k in enumerate(figs_all.keys()):
    if (i + 1) % 50 == 0:
        print(i, figs_all[k])
        shutil.copy2(figs_all[k], result_dir)