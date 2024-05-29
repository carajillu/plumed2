import glob
import pymol
from pymol import cmd

# Define color lists for clusters and false positives
cluster_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'lime']
fp_colors = ['salmon', 'forest', 'slate', 'wheat', 'violet', 'marine', 'olive', 'teal', 'chocolate', 'silver']

# Load the protein file and set its representation
cmd.load('protein.gro')
cmd.show('surface', 'protein')
cmd.set('surface_color', 'grey70', 'protein')
cmd.set('transparency', 0.5, 'protein')

# Load and color each cluster file
for i, file in enumerate(glob.glob('cluster*.pdb')):
    obj_name = 'cluster_{}'.format(i)
    cmd.load(file, obj_name)
    color = cluster_colors[i % len(cluster_colors)]
    cmd.color(color, obj_name)

# Load and color each false positive file
for i, file in enumerate(glob.glob('falsepositive*.pdb')):
    obj_name = 'falsepositive_{}'.format(i)
    cmd.load(file, obj_name)
    color = fp_colors[i % len(fp_colors)]
    cmd.color(color, obj_name)

# Redraw the scene
cmd.rebuild()
