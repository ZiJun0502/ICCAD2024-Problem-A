import re
import numpy as np
import datetime
from subprocess import check_output
from collections import defaultdict

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)
# pi/po/edges/levels/latches
def abc_stats(design_file, abc_binary, stats):    
    abc_command = "read_verilog " + design_file + "; print_stats"
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
                stats['n_input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
                stats['n_output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())
        
                ob = re.search(r'edge *= *[0-9]+', line)
                stats['n_edges'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lev *= *[0-9]+', line)
                stats['n_levels'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lat *= *[0-9]+', line)
                stats['n_latches'] = int(ob.group().split('=')[1].strip())
    except Exception as e:
        print(e)
        return None

# ands/nots/ors
def gate_stats(design_file, stats):
    with open(design_file, 'r') as file:
        lines = file.readlines()
    
    stats['n_ands'] = 0
    stats['n_ors'] = 0
    stats['n_nots'] = 0
    stats['n_nodes'] = 0
    
    for line in lines:
        if '&' in line:
            stats['n_ands'] += 1
            stats['n_nodes'] += 1
        elif '|' in line:
            stats['n_ors'] += 1
            stats['n_nodes'] += 1
        elif '~' in line:
            cnt = line.count('~')
            stats['n_nots'] += cnt
            stats['n_nodes'] += cnt

def extract_features(design_file, abc_binary='abc'):
    '''
    Returns features of a given circuit as a tuple.
    Features are listed below
    '''
    stats = {}
    gate_stats(design_file, stats)
    abc_stats(design_file, abc_binary, stats)

    return np.array([stats['n_input_pins'], stats['n_output_pins'], \
        stats['n_nodes'], stats['n_edges'], \
            stats['n_levels'], stats['n_latches'], \
                stats['n_ands'], stats['n_ors'], stats['n_nots']])