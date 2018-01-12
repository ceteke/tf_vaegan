def save_pc(pc_array, f_name):
  with open(f_name, 'w') as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1024\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1024\nDATA ascii\n")
    for i in range(len(pc_array)):
      f.write('{} {} {}\n'.format(pc_array[i][0][0], pc_array[i][1][0], pc_array[i][2][0]))