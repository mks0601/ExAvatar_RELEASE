import os
import os.path as osp

src_path = './code_to_copy'

# DECA
cmd = 'cp ' + osp.join(src_path, 'run_deca.py') + ' ./DECA/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'deca.py') + ' ./DECA/decalib/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'datasets', 'datasets.py') + ' ./DECA/decalib/datasets/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'demos', 'demo_reconstruct.py') + ' ./DECA/demos/.'
os.system(cmd)

# Hand4Whole
cmd = 'cp ' + osp.join(src_path, 'run_hand4whole.py') + ' ./Hand4Whole_RELEASE/demo/.'
os.system(cmd)

# mmpose
cmd = 'cp ' + osp.join(src_path, 'run_mmpose.py') + ' ./mmpose/.'
os.system(cmd)

# segment-anything
cmd = 'cp ' + osp.join(src_path, 'run_sam.py') + ' ./segment-anything/.'
os.system(cmd)

# Depth-Anything-V2
cmd = 'cp ' + osp.join(src_path, 'run_depth_anything.py') + ' ./Depth-Anything-V2/.'
os.system(cmd)

# COLMAP
os.makedirs('./COLMAP', exist_ok=True)
cmd = 'cp ' + osp.join(src_path, 'run_colmap.py') + ' ./COLMAP/.'
os.system(cmd)
